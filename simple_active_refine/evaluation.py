"""
知識グラフ改善プロセスの評価モジュール

各iterationでの対象トリプルのスコア変化、追加トリプル数、
知識グラフ埋め込み全体の精度（Hits@k、MRR等）を評価し、
最終的な統合レポートを生成する。
"""

import json
import os
import importlib
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    sns = importlib.import_module("seaborn")
except ModuleNotFoundError:  # pragma: no cover
    sns = None
from dataclasses import dataclass, asdict

from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


@dataclass
class IterationMetrics:
    """単一iterationの評価指標"""
    iteration: int
    
    # トリプル数
    n_triples_before: int
    n_triples_after: int
    n_triples_added: int
    
    # 対象トリプルのスコア
    target_score_before: float
    target_score_after: float
    target_score_change: float
    
    # 知識グラフ埋め込み全体の精度
    hits_at_1_before: float
    hits_at_3_before: float
    hits_at_10_before: float
    mrr_before: float
    
    hits_at_1_after: float
    hits_at_3_after: float
    hits_at_10_after: float
    mrr_after: float
    
    # 変化量
    hits_at_1_change: float
    hits_at_3_change: float
    hits_at_10_change: float
    mrr_change: float
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return asdict(self)


class IterationEvaluator:
    """Iteration評価クラス
    
    各iterationでの改善効果を定量的に評価し、
    最終的な統合レポートを生成する。
    """
    
    def __init__(self):
        """初期化"""
        self.metrics_history: List[IterationMetrics] = []
    
    def evaluate_iteration(
        self,
        iteration: int,
        kge_before: KnowledgeGraphEmbedding,
        kge_after: KnowledgeGraphEmbedding,
        target_triples: List[Tuple[str, str, str]],
        n_triples_added: int,
        dir_save: Optional[str] = None
    ) -> IterationMetrics:
        """単一iterationの評価
        
        Args:
            iteration: iteration番号
            kge_before: トリプル追加前の埋め込みモデル
            kge_after: トリプル追加後の埋め込みモデル
            target_triples: 対象トリプルのリスト
            n_triples_added: 追加されたトリプル数
            dir_save: 保存先ディレクトリ（Noneの場合は保存しない）
            
        Returns:
            IterationMetrics: 評価指標
        """
        logger.info(f'Evaluating iteration {iteration}')
        
        # トリプル数
        n_triples_before = len(kge_before.triples.mapped_triples)
        n_triples_after = len(kge_after.triples.mapped_triples)
        
        # 対象トリプルのスコア計算
        logger.info(f'  Calculating scores for {len(target_triples)} target triples')
        scores_before = kge_before.score_triples(target_triples)
        scores_after = kge_after.score_triples(target_triples)
        
        # リストの場合はnumpy配列に変換
        if isinstance(scores_before, list):
            scores_before = np.array(scores_before)
        if isinstance(scores_after, list):
            scores_after = np.array(scores_after)
        
        target_score_before = float(scores_before.mean()) if len(scores_before) > 0 else 0.0
        target_score_after = float(scores_after.mean()) if len(scores_after) > 0 else 0.0
        target_score_change = target_score_after - target_score_before
        
        logger.info(f'  Target score: {target_score_before:.4f} -> {target_score_after:.4f} (Δ={target_score_change:+.4f})')
        
        # 知識グラフ埋め込み全体の精度評価
        logger.info('  Evaluating knowledge graph embedding (before)')
        # CPU environments (no CUDA) can easily OOM during rank-based
        # evaluation for high-dimensional models (e.g., KG-FIT). Use a
        # conservative slicing configuration for stability.
        eval_before = kge_before.evaluate(batch_size=1, slice_size=64)
        
        logger.info('  Evaluating knowledge graph embedding (after)')
        eval_after = kge_after.evaluate(batch_size=1, slice_size=64)
        
        # メトリクスの取得
        hits_at_1_before = eval_before.get('hits_at_1', 0.0)
        hits_at_3_before = eval_before.get('hits_at_3', 0.0)
        hits_at_10_before = eval_before.get('hits_at_10', 0.0)
        mrr_before = eval_before.get('mean_reciprocal_rank', 0.0)
        
        hits_at_1_after = eval_after.get('hits_at_1', 0.0)
        hits_at_3_after = eval_after.get('hits_at_3', 0.0)
        hits_at_10_after = eval_after.get('hits_at_10', 0.0)
        mrr_after = eval_after.get('mean_reciprocal_rank', 0.0)
        
        # 変化量
        hits_at_1_change = hits_at_1_after - hits_at_1_before
        hits_at_3_change = hits_at_3_after - hits_at_3_before
        hits_at_10_change = hits_at_10_after - hits_at_10_before
        mrr_change = mrr_after - mrr_before
        
        logger.info(f'  Hits@1:  {hits_at_1_before:.4f} -> {hits_at_1_after:.4f} (Δ={hits_at_1_change:+.4f})')
        logger.info(f'  Hits@3:  {hits_at_3_before:.4f} -> {hits_at_3_after:.4f} (Δ={hits_at_3_change:+.4f})')
        logger.info(f'  Hits@10: {hits_at_10_before:.4f} -> {hits_at_10_after:.4f} (Δ={hits_at_10_change:+.4f})')
        logger.info(f'  MRR:     {mrr_before:.4f} -> {mrr_after:.4f} (Δ={mrr_change:+.4f})')
        
        # メトリクスオブジェクト作成
        metrics = IterationMetrics(
            iteration=iteration,
            n_triples_before=n_triples_before,
            n_triples_after=n_triples_after,
            n_triples_added=n_triples_added,
            target_score_before=target_score_before,
            target_score_after=target_score_after,
            target_score_change=target_score_change,
            hits_at_1_before=hits_at_1_before,
            hits_at_3_before=hits_at_3_before,
            hits_at_10_before=hits_at_10_before,
            mrr_before=mrr_before,
            hits_at_1_after=hits_at_1_after,
            hits_at_3_after=hits_at_3_after,
            hits_at_10_after=hits_at_10_after,
            mrr_after=mrr_after,
            hits_at_1_change=hits_at_1_change,
            hits_at_3_change=hits_at_3_change,
            hits_at_10_change=hits_at_10_change,
            mrr_change=mrr_change
        )
        
        # 履歴に追加
        self.metrics_history.append(metrics)
        
        # 保存
        if dir_save:
            self._save_iteration_report(metrics, dir_save)
        
        return metrics
    
    def _save_iteration_report(self, metrics: IterationMetrics, dir_save: str):
        """Iteration評価レポートの保存
        
        Args:
            metrics: 評価指標
            dir_save: 保存先ディレクトリ
        """
        os.makedirs(dir_save, exist_ok=True)
        
        # JSON形式で保存
        json_path = os.path.join(dir_save, 'iteration_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f'  Saved iteration metrics to: {json_path}')
        
        # Markdown形式で保存
        md_path = os.path.join(dir_save, 'iteration_evaluation.md')
        with open(md_path, 'w') as f:
            f.write(f"# Iteration {metrics.iteration} Evaluation\n\n")
            
            f.write("## Triples Added\n\n")
            f.write(f"- **Before**: {metrics.n_triples_before:,} triples\n")
            f.write(f"- **After**: {metrics.n_triples_after:,} triples\n")
            f.write(f"- **Added**: {metrics.n_triples_added:,} triples\n\n")
            
            f.write("## Target Triples Score\n\n")
            f.write(f"- **Before**: {metrics.target_score_before:.4f}\n")
            f.write(f"- **After**: {metrics.target_score_after:.4f}\n")
            f.write(f"- **Change**: {metrics.target_score_change:+.4f}\n\n")
            
            f.write("## Knowledge Graph Embedding Metrics\n\n")
            f.write("| Metric | Before | After | Change |\n")
            f.write("|--------|--------|-------|--------|\n")
            f.write(f"| Hits@1 | {metrics.hits_at_1_before:.4f} | {metrics.hits_at_1_after:.4f} | {metrics.hits_at_1_change:+.4f} |\n")
            f.write(f"| Hits@3 | {metrics.hits_at_3_before:.4f} | {metrics.hits_at_3_after:.4f} | {metrics.hits_at_3_change:+.4f} |\n")
            f.write(f"| Hits@10 | {metrics.hits_at_10_before:.4f} | {metrics.hits_at_10_after:.4f} | {metrics.hits_at_10_change:+.4f} |\n")
            f.write(f"| MRR | {metrics.mrr_before:.4f} | {metrics.mrr_after:.4f} | {metrics.mrr_change:+.4f} |\n")
        
        logger.info(f'  Saved iteration evaluation report to: {md_path}')
    
    def create_final_report(self, dir_save: str):
        """最終統合レポートの生成
        
        全iterationの評価結果をまとめ、グラフと共にMarkdownレポートを生成する。
        
        Args:
            dir_save: 保存先ディレクトリ
        """
        logger.info('Creating final evaluation report')
        
        if len(self.metrics_history) == 0:
            logger.warning('No metrics history available')
            return
        
        os.makedirs(dir_save, exist_ok=True)
        
        # DataFrameに変換
        df = pd.DataFrame([m.to_dict() for m in self.metrics_history])
        
        # CSV保存
        csv_path = os.path.join(dir_save, 'metrics_history.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f'  Saved metrics history to: {csv_path}')
        
        # グラフ生成
        self._create_plots(df, dir_save)
        
        # Markdownレポート生成
        self._create_markdown_report(df, dir_save)
    
    def _create_plots(self, df: pd.DataFrame, dir_save: str):
        """評価グラフの生成
        
        Args:
            df: メトリクス履歴のDataFrame
            dir_save: 保存先ディレクトリ
        """
        if sns is None:
            logger.warning("seaborn is not installed; skipping plot generation")
            return

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 追加トリプル数と対象トリプルスコアの関係
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        cumulative_added = df['n_triples_added'].cumsum()
        ax1.plot(df['iteration'], cumulative_added, 'o-', color='steelblue', linewidth=2, markersize=8, label='Cumulative Added Triples')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Cumulative Added Triples', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.grid(True, alpha=0.3)
        
        ax1_twin.plot(df['iteration'], df['target_score_after'], 's-', color='coral', linewidth=2, markersize=8, label='Target Score')
        ax1_twin.set_ylabel('Target Triples Score', fontsize=12, color='coral')
        ax1_twin.tick_params(axis='y', labelcolor='coral')
        
        ax1.set_title('Added Triples vs Target Score', fontsize=14, fontweight='bold')
        
        # 2. 対象トリプルのスコア変化
        ax2 = axes[0, 1]
        ax2.plot(df['iteration'], df['target_score_before'], 'o--', label='Before', linewidth=2, markersize=8)
        ax2.plot(df['iteration'], df['target_score_after'], 's-', label='After', linewidth=2, markersize=8)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Target Triples Score', fontsize=12)
        ax2.set_title('Target Triples Score Progress', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Hits@kの変化
        ax3 = axes[1, 0]
        ax3.plot(df['iteration'], df['hits_at_1_after'], 'o-', label='Hits@1', linewidth=2, markersize=8)
        ax3.plot(df['iteration'], df['hits_at_3_after'], 's-', label='Hits@3', linewidth=2, markersize=8)
        ax3.plot(df['iteration'], df['hits_at_10_after'], '^-', label='Hits@10', linewidth=2, markersize=8)
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Hits@k', fontsize=12)
        ax3.set_title('Knowledge Graph Embedding: Hits@k', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. MRRの変化
        ax4 = axes[1, 1]
        ax4.plot(df['iteration'], df['mrr_after'], 'o-', color='purple', linewidth=2, markersize=8)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('MRR', fontsize=12)
        ax4.set_title('Knowledge Graph Embedding: MRR', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        plot_path = os.path.join(dir_save, 'evaluation_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'  Saved evaluation plots to: {plot_path}')
    
    def _create_markdown_report(self, df: pd.DataFrame, dir_save: str):
        """Markdownレポートの生成
        
        Args:
            df: メトリクス履歴のDataFrame
            dir_save: 保存先ディレクトリ
        """
        md_path = os.path.join(dir_save, 'final_evaluation_report.md')
        
        with open(md_path, 'w') as f:
            f.write("# Final Evaluation Report\n\n")
            f.write("This report summarizes the knowledge graph improvement process across all iterations.\n\n")
            
            # サマリー統計
            f.write("## Summary Statistics\n\n")
            
            total_added = df['n_triples_added'].sum()
            initial_triples = df.iloc[0]['n_triples_before']
            final_triples = df.iloc[-1]['n_triples_after']
            
            f.write(f"- **Total iterations**: {len(df)}\n")
            f.write(f"- **Initial triples**: {initial_triples:,}\n")
            f.write(f"- **Final triples**: {final_triples:,}\n")
            f.write(f"- **Total added triples**: {total_added:,}\n")
            f.write(f"- **Growth rate**: {(total_added/initial_triples)*100:.2f}%\n\n")
            
            # 対象トリプルスコアの改善
            f.write("## Target Triples Score Improvement\n\n")
            
            initial_score = df.iloc[0]['target_score_before']
            final_score = df.iloc[-1]['target_score_after']
            total_change = final_score - initial_score
            
            f.write(f"- **Initial score**: {initial_score:.4f}\n")
            f.write(f"- **Final score**: {final_score:.4f}\n")
            f.write(f"- **Total change**: {total_change:+.4f}\n")
            f.write(f"- **Relative improvement**: {(total_change/abs(initial_score))*100:+.2f}%\n\n")
            
            # 知識グラフ埋め込み精度の改善
            f.write("## Knowledge Graph Embedding Improvement\n\n")
            
            f.write("| Metric | Initial | Final | Change | Relative Change |\n")
            f.write("|--------|---------|-------|--------|----------------|\n")
            
            for metric, name in [
                ('hits_at_1', 'Hits@1'),
                ('hits_at_3', 'Hits@3'),
                ('hits_at_10', 'Hits@10'),
                ('mrr', 'MRR')
            ]:
                initial = df.iloc[0][f'{metric}_before']
                final = df.iloc[-1][f'{metric}_after']
                change = final - initial
                rel_change = (change / abs(initial) * 100) if initial != 0 else 0
                
                f.write(f"| {name} | {initial:.4f} | {final:.4f} | {change:+.4f} | {rel_change:+.2f}% |\n")
            
            f.write("\n")
            
            # Iteration別の詳細
            f.write("## Iteration-by-Iteration Details\n\n")
            
            f.write("| Iter | Added Triples | Target Score Δ | Hits@1 Δ | Hits@10 Δ | MRR Δ |\n")
            f.write("|------|---------------|----------------|----------|-----------|--------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['iteration']} | {row['n_triples_added']:,} | "
                       f"{row['target_score_change']:+.4f} | "
                       f"{row['hits_at_1_change']:+.4f} | "
                       f"{row['hits_at_10_change']:+.4f} | "
                       f"{row['mrr_change']:+.4f} |\n")
            
            f.write("\n")
            
            # グラフの挿入
            f.write("## Visualization\n\n")
            f.write("![Evaluation Summary](evaluation_summary.png)\n\n")
            
            # 結論
            f.write("## Conclusion\n\n")
            
            if total_change > 0:
                f.write(f"The iterative knowledge graph improvement process successfully increased "
                       f"the target triples score by {total_change:.4f} ({(total_change/abs(initial_score))*100:+.2f}%) "
                       f"while adding {total_added:,} new triples to the knowledge graph.\n\n")
            else:
                f.write(f"The target triples score decreased by {abs(total_change):.4f} "
                       f"({(total_change/abs(initial_score))*100:.2f}%) during the process. "
                       f"This may indicate that the added triples introduced noise or that "
                       f"the evaluation metrics need adjustment.\n\n")
            
            # 各メトリクスの改善状況
            hits1_improved = df.iloc[-1]['hits_at_1_after'] > df.iloc[0]['hits_at_1_before']
            hits10_improved = df.iloc[-1]['hits_at_10_after'] > df.iloc[0]['hits_at_10_before']
            mrr_improved = df.iloc[-1]['mrr_after'] > df.iloc[0]['mrr_before']
            
            improved_count = sum([hits1_improved, hits10_improved, mrr_improved])
            
            if improved_count >= 2:
                f.write("The knowledge graph embedding quality improved across most metrics, "
                       "indicating successful knowledge augmentation.\n")
            else:
                f.write("The knowledge graph embedding quality showed mixed results. "
                       "Further analysis of the added triples may be necessary.\n")
        
        logger.info(f'  Saved final evaluation report to: {md_path}')
