"""
ルール選択戦略モジュール

LLMベースのポリシー駆動型ルール選択機能を提供する。
統計的な獲得関数の代わりに、LLMが履歴情報を元に
自然言語でポリシーを更新し、それに基づいてルールを選択する。

設計原則:
- Google Style Docstringに従う
- PEP8準拠のコーディング
- 型ヒントを明示
- util.get_logger()を使用した統一的なログ出力
"""

from __future__ import annotations
import os
import random
import math
import uuid
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

from settings import OPENAI_API_KEY
from simple_active_refine.amie import AmieRule
from simple_active_refine.rule_history import RuleHistory, RuleStatistics
from simple_active_refine.util import get_logger

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logger = get_logger('rule_selector')


@dataclass
class RuleWithId:
    """ID付きルール
    
    Attributes:
        rule_id: ルールの一意識別子
        rule: AmieRuleオブジェクト
    """
    rule_id: str
    rule: AmieRule
    
    @classmethod
    def create(cls, rule: AmieRule, rule_id: Optional[str] = None) -> RuleWithId:
        """RuleWithIdインスタンスを生成
        
        Args:
            rule: AmieRuleオブジェクト
            rule_id: ルールID（Noneの場合は自動生成）
            
        Returns:
            RuleWithId: 生成されたインスタンス
        """
        if rule_id is None:
            rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        return cls(rule_id=rule_id, rule=rule)


class SelectionPolicy(BaseModel):
    """ルール選択ポリシー（構造化出力用）"""
    reasoning: str = Field(..., description="ポリシー更新の理由と現在の状況分析")
    policy_text: str = Field(..., description="今後のルール選択の方針を自然言語で記述")
    selected_rule_ids: List[str] = Field(..., description="選択するルールのIDリスト")
    rationale_per_rule: Dict[str, str] = Field(..., description="各ルールを選択した理由")


class RuleSelector:
    """ルール選択の基底クラス"""
    
    def __init__(self, history: Optional[RuleHistory] = None):
        """RuleSelectorの初期化
        
        Args:
            history: ルール履歴オブジェクト
        """
        self.history = history or RuleHistory()
    
    def select_rules(self, 
                     rule_pool: List[RuleWithId], 
                     k: int = 3,
                     iteration: int = 0) -> tuple[List[RuleWithId], Optional[str]]:
        """ルールpoolからk個のルールを選択
        
        Args:
            rule_pool: 選択候補のルール集合
            k: 選択するルール数
            iteration: 現在のiteration番号
        
        Returns:
            tuple[List[RuleWithId], Optional[str]]: (選択されたk個のルール, ポリシーテキスト)
        """
        raise NotImplementedError


class LLMPolicyRuleSelector(RuleSelector):
    """LLMベースのポリシー駆動型ルール選択
    
    履歴情報を元にLLMがポリシーを自然言語で記述・更新し、
    そのポリシーに基づいてルールを選択する。
    
    ポリシーには以下の情報を含む:
    - 各ルールのスコア改善統計（平均、成功率、直近の傾向）
    - これまでの試行回数
    - 選択すべきルールの特徴
    - 探索と活用のバランス
    """
    
    def __init__(self,
                 history: Optional[RuleHistory] = None,
                 chat_model: str = "gpt-4o",
                 temperature: float = 0.3):
        """LLMPolicyRuleSelectorの初期化
        
        Args:
            history: ルール履歴
            chat_model: 使用するLLMモデル名
            temperature: LLMのtemperature（0.3でやや創造的）
        """
        super().__init__(history)
        self.llm = ChatOpenAI(model_name=chat_model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(SelectionPolicy)
        self.current_policy: Optional[str] = None
    
    def _format_rule_statistics(self, rule_pool: List[RuleWithId]) -> str:
        """ルールpoolの統計情報を整形
        
        Args:
            rule_pool: ルールpool
            
        Returns:
            str: 整形された統計情報
        """
        all_stats = self.history.get_all_rule_statistics()
        
        # デバッグ: 履歴の内容を確認
        logger.info(f"History has {len(self.history.records)} total records")
        logger.info(f"History tracked {len(all_stats)} rules with statistics")
        
        lines = []
        lines.append("## Current Rule Pool Statistics\n")
        
        for i, rule_with_id in enumerate(rule_pool, 1):
            rule_id = rule_with_id.rule_id
            rule = rule_with_id.rule
            
            # ルールの内容
            body_str = ' ∧ '.join([f"({tp.s} {tp.p} {tp.o})" for tp in rule.body])
            head_str = f"({rule.head.s} {rule.head.p} {rule.head.o})"
            
            lines.append(f"### Rule {i}: {rule_id}")
            lines.append(f"**Pattern**: {body_str} ⇒ {head_str}")
            
            # 統計情報
            if rule_id in all_stats:
                stats = all_stats[rule_id]
                lines.append(f"- **Trials**: {stats.total_iterations}")
                lines.append(f"- **Mean score change**: {stats.mean_score_change:.6f}")
                lines.append(f"- **Std deviation**: {stats.std_score_change:.6f}")
                lines.append(f"- **Success rate**: {stats.success_rate:.2%}")
                lines.append(f"- **Recent performance** (last 3): {stats.recent_performance:.6f}")
                lines.append(f"- **Total triples added**: {stats.total_triples_added}")
            else:
                lines.append(f"- **Trials**: 0 (UNTRIED)")
                lines.append(f"- **Status**: This rule has never been tested")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_selection_prompt(self,
                                 rule_pool: List[RuleWithId],
                                 k: int,
                                 iteration: int) -> str:
        """ルール選択用のプロンプトを生成
        
        Args:
            rule_pool: ルールpool
            k: 選択するルール数
            iteration: 現在のiteration番号
            
        Returns:
            str: プロンプト
        """
        stats_text = self._format_rule_statistics(rule_pool)
        
        prompt = f"""You are an expert in knowledge graph refinement using reinforcement learning principles.

Your task is to select {k} rules from the rule pool for iteration {iteration}.

{stats_text}

## Current Selection Policy
{self.current_policy if self.current_policy else "No policy yet (first iteration). Start with exploration."}

## Your Task

1. **Analyze** the statistics above carefully:
   - Which rules have shown consistent improvement?
   - Which rules are untried or undertried?
   - What are the recent trends?

2. **Update the policy**:
   - Consider the trade-off between exploration (trying new/untried rules) and exploitation (using proven rules)
   - Early iterations should focus more on exploration
   - Later iterations can focus more on exploitation of proven rules
   - But always keep some exploration to avoid local optima

3. **Select {k} rules**:
   - Return the rule IDs in the `selected_rule_ids` field
   - Provide rationale for each selection in `rationale_per_rule`

## Guidelines

- **Exploration**: Prioritize untried rules or rules with few trials to gather more information
- **Exploitation**: Favor rules with high mean score change and success rate
- **Recency**: Give weight to recent performance trends
- **Diversity**: Consider selecting rules with different patterns to cover various scenarios
- **Iteration stage**: Early iterations (1-3) should be more exploratory; later iterations can be more exploitative

## Output Format

Provide a JSON with:
- `reasoning`: Your analysis of the current situation
- `policy_text`: Updated policy for future iterations (natural language)
- `selected_rule_ids`: List of {k} rule IDs to select
- `rationale_per_rule`: Dictionary mapping each selected rule ID to selection rationale
"""
        
        return prompt
    
    def select_rules(self,
                     rule_pool: List[RuleWithId],
                     k: int = 3,
                     iteration: int = 0) -> tuple[List[RuleWithId], Optional[str]]:
        """LLMポリシーに基づいてk個のルールを選択
        
        Args:
            rule_pool: 選択候補のルール集合
            k: 選択するルール数
            iteration: 現在のiteration番号
            
        Returns:
            tuple[List[RuleWithId], Optional[str]]: (選択されたルール, 更新されたポリシー)
        """
        if len(rule_pool) <= k:
            logger.info(f"Pool size ({len(rule_pool)}) <= k ({k}), selecting all rules")
            return rule_pool, self.current_policy
        
        # iteration=0の場合は、初期ルールプール構築時の並び順を尊重して先頭から選択
        if iteration == 0:
            logger.info(
                "Iteration 0: Selecting first %d rules in the existing pool order "
                "(aligned with initial pool construction criteria)",
                k,
            )

            selected = rule_pool[:k]

            for i, rule_with_id in enumerate(selected, 1):
                logger.info("  Selected: %s (rank %d in initial pool)", rule_with_id.rule_id, i)

            # 初期ポリシーを設定
            if self.current_policy is None:
                self.current_policy = (
                    "Initial iteration: Used the pre-ranked rule pool (same ordering as pool construction). "
                    "Later iterations adapt based on observed performance while keeping exploration/exploitation balance."
                )

            return selected, self.current_policy
        
        # プロンプト生成
        prompt = self._create_selection_prompt(rule_pool, k, iteration)
        
        logger.info(f"Consulting LLM for rule selection (iteration {iteration})")
        logger.debug(f"Prompt: {prompt}")
        
        # LLMに問い合わせ
        try:
            result: SelectionPolicy = self.structured_llm.invoke(prompt)
            
            # ポリシー更新
            self.current_policy = result.policy_text
            
            logger.info(f"LLM Reasoning: {result.reasoning}")
            logger.info(f"Updated Policy: {result.policy_text}")
            
            # 選択されたルールを取得
            rule_id_to_rule = {r.rule_id: r for r in rule_pool}
            selected = []
            
            for rule_id in result.selected_rule_ids:
                if rule_id in rule_id_to_rule:
                    selected.append(rule_id_to_rule[rule_id])
                    rationale = result.rationale_per_rule.get(rule_id, "No rationale provided")
                    logger.info(f"  Selected: {rule_id} - {rationale}")
                else:
                    logger.warning(f"  Rule {rule_id} not found in pool, skipping")
            
            # 選択数が不足している場合はランダムに補完
            if len(selected) < k:
                remaining = [r for r in rule_pool if r not in selected]
                additional = random.sample(remaining, min(k - len(selected), len(remaining)))
                selected.extend(additional)
                logger.warning(f"Added {len(additional)} random rules to reach k={k}")
            
            # 選択数が超過している場合は切り詰め
            selected = selected[:k]
            
            return selected, self.current_policy
            
        except Exception as e:
            logger.error(f"LLM selection failed: {e}")
            logger.warning("Falling back to random selection")
            selected = random.sample(rule_pool, k)
            return selected, self.current_policy
    
    def save_policy(self, filepath: str) -> None:
        """現在のポリシーをファイルに保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        if self.current_policy:
            with open(filepath, 'w') as f:
                f.write(self.current_policy)
            logger.info(f"Saved policy to {filepath}")
    
    def load_policy(self, filepath: str) -> None:
        """ポリシーをファイルから読み込み
        
        Args:
            filepath: 読み込み元ファイルパス
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.current_policy = f.read()
            logger.info(f"Loaded policy from {filepath}")


class UCBRuleSelector(RuleSelector):
    """UCB (Upper Confidence Bound) アルゴリズムによるルール選択
    
    探索と活用のバランスを取りながらルールを選択する。
    スコア向上の期待値が高く、かつ不確実性が高いルールを優先的に選択。
    
    UCB = mean_reward + c * sqrt(ln(total_iterations) / n_trials)
    """
    
    def __init__(self, 
                 history: Optional[RuleHistory] = None,
                 exploration_param: float = 1.0):
        """UCBRuleSelectorの初期化
        
        Args:
            history: ルール履歴
            exploration_param: 探索パラメータ（大きいほど未知のルールを選びやすい）
        """
        super().__init__(history)
        self.c = exploration_param
    
    def _calculate_ucb_score(self, 
                            rule_id: str, 
                            total_iterations: int,
                            stats: Optional[RuleStatistics]) -> float:
        """UCBスコアを計算
        
        Args:
            rule_id: ルールID
            total_iterations: 全体のiteration数
            stats: ルールの統計情報
        
        Returns:
            float: UCBスコア
        """
        if stats is None or stats.total_iterations == 0:
            # 未試行のルールには無限大のスコアを与える（必ず選択される）
            return float('inf')
        
        # 平均報酬（スコア変化の平均）
        mean_reward = stats.mean_score_change
        
        # 探索ボーナス
        exploration_bonus = self.c * math.sqrt(
            math.log(total_iterations + 1) / stats.total_iterations
        )
        
        ucb_score = mean_reward + exploration_bonus
        
        return ucb_score
    
    def select_rules(self, 
                     rule_pool: List[RuleWithId], 
                     k: int = 3,
                     iteration: int = 0) -> tuple[List[RuleWithId], Optional[str]]:
        """UCBスコアに基づいてk個のルールを選択
        
        Args:
            rule_pool: 選択候補のルール集合
            k: 選択するルール数
            iteration: 現在のiteration番号
            
        Returns:
            tuple[List[RuleWithId], Optional[str]]: (選択されたk個のルール, None)
        """
        
        if len(rule_pool) <= k:
            logger.info(f"Pool size ({len(rule_pool)}) <= k ({k}), selecting all rules")
            return rule_pool, None
        
        # 各ルールの統計情報とUCBスコアを計算
        rule_scores = []
        all_stats = self.history.get_all_rule_statistics()
        
        for rule_with_id in rule_pool:
            rule_id = rule_with_id.rule_id
            stats = all_stats.get(rule_id)
            
            ucb_score = self._calculate_ucb_score(rule_id, iteration, stats)
            rule_scores.append((rule_with_id, ucb_score, stats))
            
            logger.debug(f"Rule {rule_id}: UCB={ucb_score:.6f}, "
                        f"mean_Δ={stats.mean_score_change if stats else 'N/A'}, "
                        f"n_trials={stats.total_iterations if stats else 0}")
        
        # UCBスコアでソートしてtop-kを選択
        rule_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [rs[0] for rs in rule_scores[:k]]
        
        logger.info(f"Selected {k} rules using UCB (c={self.c}):")
        for i, (rule_with_id, score, stats) in enumerate(rule_scores[:k], 1):
            logger.info(f"  {i}. {rule_with_id.rule_id}: UCB={score:.6f}")
        
        return selected, None


class EpsilonGreedyRuleSelector(RuleSelector):
    """ε-greedy による ルール選択
    
    確率εで探索（ランダム選択）、1-εで活用（best選択）。
    """
    
    def __init__(self, 
                 history: Optional[RuleHistory] = None,
                 epsilon: float = 0.1):
        """EpsilonGreedyRuleSelectorの初期化
        
        Args:
            history: ルール履歴
            epsilon: 探索確率（0-1の範囲）
        """
        super().__init__(history)
        self.epsilon = epsilon
    
    def select_rules(self, 
                     rule_pool: List[RuleWithId], 
                     k: int = 3,
                     iteration: int = 0) -> tuple[List[RuleWithId], Optional[str]]:
        """ε-greedyでk個のルールを選択
        
        Args:
            rule_pool: 選択候補のルール集合
            k: 選択するルール数
            iteration: 現在のiteration番号
            
        Returns:
            tuple[List[RuleWithId], Optional[str]]: (選択されたk個のルール, None)
        """
        
        if len(rule_pool) <= k:
            return rule_pool, None
        
        selected = []
        all_stats = self.history.get_all_rule_statistics()
        
        for _ in range(k):
            if random.random() < self.epsilon:
                # 探索: ランダム選択
                available = [r for r in rule_pool if r not in selected]
                if available:
                    selected.append(random.choice(available))
            else:
                # 活用: 最良のルールを選択
                available = [r for r in rule_pool if r not in selected]
                if not available:
                    break
                
                best_rule = None
                best_score = float('-inf')
                
                for rule_with_id in available:
                    stats = all_stats.get(rule_with_id.rule_id)
                    if stats is None:
                        score = 0.0  # 未試行のルールは中立的なスコア
                    else:
                        score = stats.mean_score_change
                    
                    if score > best_score:
                        best_score = score
                        best_rule = rule_with_id
                
                if best_rule:
                    selected.append(best_rule)
        
        logger.info(f"Selected {len(selected)} rules using ε-greedy (ε={self.epsilon})")
        return selected, None


class RandomRuleSelector(RuleSelector):
    """単純ランダムでルールを選択するセレクタ"""

    def select_rules(
        self,
        rule_pool: List[RuleWithId],
        k: int = 3,
        iteration: int = 0,
    ) -> tuple[List[RuleWithId], Optional[str]]:
        if len(rule_pool) <= k:
            return rule_pool, None

        selected = random.sample(rule_pool, k)
        logger.info("Selected %d rules randomly", len(selected))
        return selected, None


def create_rule_selector(strategy: str = 'llm_policy', 
                        history: Optional[RuleHistory] = None,
                        **kwargs) -> RuleSelector:
    """ルール選択戦略のファクトリー関数
    
    Args:
        strategy: 戦略名 ('llm_policy', 'ucb', 'epsilon_greedy')
        history: ルール履歴
        **kwargs: 各戦略固有のパラメータ
    
    Returns:
        RuleSelector: RuleSelectorインスタンス
        
    Raises:
        ValueError: 未知の戦略名が指定された場合
    """
    if strategy == 'llm_policy':
        chat_model = kwargs.get('chat_model', 'gpt-4o')
        temperature = kwargs.get('temperature', 0.3)
        return LLMPolicyRuleSelector(history, chat_model, temperature)
    
    elif strategy == 'ucb':
        exploration_param = kwargs.get('exploration_param', 1.0)
        return UCBRuleSelector(history, exploration_param)
    
    elif strategy == 'epsilon_greedy':
        epsilon = kwargs.get('epsilon', 0.1)
        return EpsilonGreedyRuleSelector(history, epsilon)

    elif strategy == 'random':
        return RandomRuleSelector(history)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Supported strategies: 'llm_policy', 'ucb', 'epsilon_greedy'")
