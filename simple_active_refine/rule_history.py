"""
ルール履歴管理モジュール

各iterationで各ルールがどのような効果を持ったかを記録し、
履歴全体からルールの有効性を評価する機能を提供する。

設計原則:
- Google Style Docstringに従う
- PEP8準拠のコーディング
- 型ヒントを明示
- util.get_logger()を使用した統一的なログ出力
"""

from __future__ import annotations
import os
import pickle
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import statistics

from simple_active_refine.amie import AmieRule
from simple_active_refine.util import get_logger

logger = get_logger('rule_history')

Triple = Tuple[str, str, str]


@dataclass
class RuleEvaluationRecord:
    """1回のiterationにおける1ルールの評価記録
    
    Attributes:
        iteration: iteration番号
        rule_id: ルールの一意識別子
        rule: AmieRuleオブジェクト
        target_triples: このルールが適用されたtarget tripleのリスト
        added_triples: このルールによって追加されたトリプルのリスト
        score_changes: 各target tripleのスコア変化 (after - before)
        mean_score_change: スコア変化の平均値
        std_score_change: スコア変化の標準偏差
        positive_changes: スコアが向上したトリプル数
        negative_changes: スコアが低下したトリプル数
    """
    
    iteration: int
    rule_id: str
    rule: AmieRule
    target_triples: List[Triple]
    added_triples: List[Triple]
    score_changes: List[float]
    mean_score_change: float
    std_score_change: float
    positive_changes: int
    negative_changes: int
    
    def to_dict(self) -> Dict:
        """辞書形式に変換（JSON保存用）
        
        Returns:
            Dict: JSON serializable な辞書
        """
        return {
            'iteration': self.iteration,
            'rule_id': self.rule_id,
            'rule': {
                'head': self.rule.head.to_tsv(),
                'body': [bp.to_tsv() for bp in self.rule.body]
            },
            'target_triples': [list(t) for t in self.target_triples],
            'added_triples': [list(t) for t in self.added_triples],
            'score_changes': self.score_changes,
            'mean_score_change': self.mean_score_change,
            'std_score_change': self.std_score_change,
            'positive_changes': self.positive_changes,
            'negative_changes': self.negative_changes
        }


@dataclass
class RuleStatistics:
    """ルールの統計情報
    
    Attributes:
        rule_id: ルールID
        total_iterations: 使用された総iteration数
        total_triples_added: 追加した総トリプル数
        mean_score_change: 全体の平均スコア変化
        std_score_change: スコア変化の標準偏差
        success_rate: スコア向上の割合
        recent_performance: 直近N回の平均スコア変化
    """
    
    rule_id: str
    total_iterations: int
    total_triples_added: int
    mean_score_change: float
    std_score_change: float
    success_rate: float
    recent_performance: float
    
    def __repr__(self) -> str:
        return (f"RuleStats(id={self.rule_id}, "
                f"iters={self.total_iterations}, "
                f"mean_Δ={self.mean_score_change:.4f}, "
                f"success={self.success_rate:.2%}, "
                f"recent={self.recent_performance:.4f})")


class RuleHistory:
    """全ルールの履歴管理クラス
    
    各ルールの評価記録を保存し、統計情報の計算や分析機能を提供する。
    """
    
    def __init__(self):
        """RuleHistoryの初期化"""
        self.records: List[RuleEvaluationRecord] = []
        self._records_by_rule: Dict[str, List[RuleEvaluationRecord]] = {}
        self._records_by_iteration: Dict[int, List[RuleEvaluationRecord]] = {}
    
    def add_record(self, record: RuleEvaluationRecord) -> None:
        """評価記録を追加
        
        Args:
            record: 追加する評価記録
        """
        self.records.append(record)
        
        # ルールIDによるインデックス更新
        if record.rule_id not in self._records_by_rule:
            self._records_by_rule[record.rule_id] = []
        self._records_by_rule[record.rule_id].append(record)
        
        # iterationによるインデックス更新
        if record.iteration not in self._records_by_iteration:
            self._records_by_iteration[record.iteration] = []
        self._records_by_iteration[record.iteration].append(record)
        
        logger.debug(f"Added record: iter={record.iteration}, rule={record.rule_id}, "
                    f"mean_change={record.mean_score_change:.4f}")
    
    def get_records_for_rule(self, rule_id: str) -> List[RuleEvaluationRecord]:
        """特定ルールの全履歴を取得
        
        Args:
            rule_id: ルールID
            
        Returns:
            List[RuleEvaluationRecord]: 該当ルールの評価記録リスト
        """
        return self._records_by_rule.get(rule_id, [])
    
    def get_records_for_iteration(self, iteration: int) -> List[RuleEvaluationRecord]:
        """特定iterationの全記録を取得
        
        Args:
            iteration: iteration番号
            
        Returns:
            List[RuleEvaluationRecord]: 該当iterationの評価記録リスト
        """
        return self._records_by_iteration.get(iteration, [])
    
    def get_rule_statistics(self, rule_id: str, recent_n: int = 3) -> Optional[RuleStatistics]:
        """ルールの統計情報を計算
        
        Args:
            rule_id: ルールID
            recent_n: 直近N回の平均計算に使用する回数
            
        Returns:
            Optional[RuleStatistics]: 統計情報（記録がない場合はNone）
        """
        records = self.get_records_for_rule(rule_id)
        if not records:
            return None
        
        all_score_changes = []
        total_positive = 0
        total_negative = 0
        total_triples = 0
        
        for rec in records:
            all_score_changes.extend(rec.score_changes)
            total_positive += rec.positive_changes
            total_negative += rec.negative_changes
            total_triples += len(rec.added_triples)
        
        # 直近N回の平均スコア変化
        recent_records = records[-recent_n:]
        recent_changes = []
        for rec in recent_records:
            recent_changes.extend(rec.score_changes)
        
        return RuleStatistics(
            rule_id=rule_id,
            total_iterations=len(records),
            total_triples_added=total_triples,
            mean_score_change=statistics.mean(all_score_changes) if all_score_changes else 0.0,
            std_score_change=statistics.stdev(all_score_changes) if len(all_score_changes) > 1 else 0.0,
            success_rate=total_positive / (total_positive + total_negative) 
                        if (total_positive + total_negative) > 0 else 0.0,
            recent_performance=statistics.mean(recent_changes) if recent_changes else 0.0
        )
    
    def get_all_rule_statistics(self, recent_n: int = 3) -> Dict[str, RuleStatistics]:
        """全ルールの統計情報を取得
        
        Args:
            recent_n: 直近N回の平均計算に使用する回数
            
        Returns:
            Dict[str, RuleStatistics]: ルールID -> 統計情報のマッピング
        """
        stats = {}
        for rule_id in self._records_by_rule.keys():
            stat = self.get_rule_statistics(rule_id, recent_n)
            if stat:
                stats[rule_id] = stat
        return stats
    
    def generate_summary_report(self) -> str:
        """テキストサマリーレポートを生成
        
        Returns:
            str: Markdown形式のレポート
        """
        report = "# Rule History Summary Report\n\n"
        
        stats = self.get_all_rule_statistics()
        if not stats:
            report += "No records available.\n"
            return report
        
        report += f"## Overall Statistics\n\n"
        report += f"- Total rules tracked: {len(stats)}\n"
        report += f"- Total iterations: {len(self._records_by_iteration)}\n"
        report += f"- Total records: {len(self.records)}\n\n"
        
        report += "## Rule Rankings (by mean score change)\n\n"
        sorted_stats = sorted(stats.values(), 
                             key=lambda s: s.mean_score_change, 
                             reverse=True)
        
        for i, stat in enumerate(sorted_stats, 1):
            report += f"{i}. **{stat.rule_id}**\n"
            report += f"   - Mean Δscore: {stat.mean_score_change:.6f}\n"
            report += f"   - Success rate: {stat.success_rate:.2%}\n"
            report += f"   - Recent performance: {stat.recent_performance:.6f}\n"
            report += f"   - Total iterations: {stat.total_iterations}\n"
            report += f"   - Total triples added: {stat.total_triples_added}\n\n"
        
        return report
    
    def save(self, filepath: str) -> None:
        """履歴をファイルに保存（pickle）
        
        Args:
            filepath: 保存先ファイルパス
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved rule history to {filepath}")
    
    def save_json(self, filepath: str) -> None:
        """履歴をJSON形式で保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        data = {
            'records': [rec.to_dict() for rec in self.records]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved rule history (JSON) to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> RuleHistory:
        """履歴をファイルから読み込み（pickle）
        
        Args:
            filepath: 読み込み元ファイルパス
            
        Returns:
            RuleHistory: 読み込んだ履歴オブジェクト
        """
        with open(filepath, 'rb') as f:
            history = pickle.load(f)
        logger.info(f"Loaded rule history from {filepath}")
        return history
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __repr__(self) -> str:
        return (f"RuleHistory(records={len(self.records)}, "
                f"rules={len(self._records_by_rule)}, "
                f"iterations={len(self._records_by_iteration)})")
