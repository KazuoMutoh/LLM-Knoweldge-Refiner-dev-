# Knowledge Graph Improvement - Improvement Report

**Date**: 2025-12-14  
**Version**: v2 (Improved)

---

## Executive Summary

テスト実験（epoch=2, n_iter=2）の結果分析に基づき、知識グラフ改善アルゴリズムに3つの重要な改善を加えました。主な問題点は、**追加されたトリプルの品質が低く、対象トリプルのスコアが大幅に悪化**（-73%）していたことです。

---

## 問題点の分析

### テスト実験の結果

| Metric | Iteration 1 | Iteration 2 | Total Change |
|--------|-------------|-------------|--------------|
| Target Score | -4.74 | -1.96 | **-6.71 (-73%)** |
| Added Triples | 958 | 1,182 | 2,140 |
| Hits@1 | -0.0123 | +0.0392 | +0.0270 (+17%) |
| Hits@3 | +0.0196 | +0.0025 | +0.0221 (+6%) |
| MRR | -0.0043 | +0.0209 | +0.0166 (+6%) |

### 特定された問題

1. **低品質トリプルの混入**
   - ルールマッチングのみで追加していたため、スコアの低いトリプルが大量に含まれていた
   - 埋込空間で尤もらしくないトリプルが知識グラフに追加されていた

2. **評価指標の矛盾**
   - 対象トリプルのスコアは悪化しているのに、全体的なHits@k、MRRは改善
   - これは、追加トリプルがランダムノイズとして働き、モデルを混乱させていた可能性

3. **Early Stoppingの欠如**
   - スコアが悪化し続けても、設定されたiteration数まで実行を継続
   - 無駄な計算と更なる品質劣化を招いていた

---

## 実装した改善策

### 改善1: トリプル品質フィルタリング

**目的**: 低品質なトリプルを事前に除外し、高品質なトリプルのみを追加する

**実装内容**:
```python
def filter_triples_by_quality(
    candidate_triples: List[Tuple[str, str, str]],
    kge: KnowledgeGraphEmbedding,
    min_score_percentile: float = 40.0
) -> List[Tuple[str, str, str]]:
    """
    候補トリプルをスコアリングし、低品質なトリプルを除外する。
    
    Args:
        candidate_triples: 候補トリプルのリスト
        kge: 知識グラフ埋込モデル
        min_score_percentile: 最小スコアパーセンタイル（これ以下は除外）
        
    Returns:
        フィルタリング後のトリプルリスト
    """
    scores = kge.score_triples(candidate_triples, normalize=True, norm_method='sigmoid')
    threshold = np.percentile(scores, min_score_percentile)
    
    filtered = [triple for triple, score in zip(candidate_triples, scores) 
                if score >= threshold]
    
    return filtered
```

**期待される効果**:
- 埋込空間で尤もらしいトリプルのみが追加される
- 対象トリプルのスコア悪化を抑制
- モデルの過適合を防止

**パラメータ**:
- `--min_score_percentile`: デフォルト40（下位60%を除外）
- 調整可能：より厳しくする場合は50-70

---

### 改善2: Early Stopping機構

**目的**: 改善の見込みがない場合、早期に実験を停止して計算資源を節約

**実装内容**:
```python
def check_iteration_quality(
    target_score_history: List[float],
    patience: int = 2
) -> bool:
    """
    対象トリプルのスコアが連続して悪化している場合、改善の見込みがないと判断。
    
    Args:
        target_score_history: 各iterationの対象トリプル平均スコア履歴
        patience: 許容する連続悪化回数
        
    Returns:
        True: 継続すべき、False: 停止すべき
    """
    if len(target_score_history) < patience + 1:
        return True
    
    recent = target_score_history[-(patience+1):]
    is_degrading = all(recent[i] < recent[i-1] for i in range(1, len(recent)))
    
    if is_degrading:
        logger.warning(f'EARLY STOPPING: Target score has degraded for {patience} consecutive iterations')
        return False
    
    return True
```

**期待される効果**:
- 無駄な計算の削減（epoch=100で数時間 → 早期停止で数十分）
- 品質劣化の拡大防止
- 実験の効率化

**パラメータ**:
- `--early_stop_patience`: デフォルト2（2回連続悪化で停止）
- 調整可能：より寛容にする場合は3-4

---

### 改善3: より詳細なログ出力

**目的**: 問題の早期発見とデバッグの効率化

**実装内容**:
- トリプルフィルタリングの詳細ログ
- スコア変化の統計情報（平均、標準偏差、正/負の数）
- Early Stopping判定時の警告メッセージ

**期待される効果**:
- リアルタイムでの品質監視
- 問題発生時の迅速な対応
- 実験結果の詳細な分析

---

## 新しいコマンドラインオプション

```bash
python3 main_v2.py \
  --n_iter 3 \
  --num_epochs 100 \
  --dir ./experiments/20251214/improved_v2 \
  --min_score_percentile 40.0 \
  --early_stop_patience 2
```

**オプション**:
- `--min_score_percentile`: トリプル品質フィルタリング閾値（0-100、デフォルト40）
- `--early_stop_patience`: Early Stopping許容回数（1-10、デフォルト2）

---

## 予想される結果

### 改善前（オリジナル版）
- 対象トリプルスコア: **-73%悪化**
- 追加トリプル数: 2,140個（品質不明）
- Hits@1: +17%
- MRR: +6%

### 改善後（v2版）の期待
- 対象トリプルスコア: **+10-20%改善**（または-10%以内の悪化）
- 追加トリプル数: 500-1,000個（高品質なトリプルのみ）
- Hits@1: +20-30%（より明確な改善）
- MRR: +10-15%（より明確な改善）
- 実行時間: Early Stoppingにより短縮される可能性

---

## アルゴリズムの変更点まとめ

```
【オリジナル版】
ルールマッチング → トリプル追加 → 埋込学習 → 評価

【改善版v2】
ルールマッチング → 品質フィルタリング → トリプル追加 → 埋込学習 → 評価 → Early Stopping判定
                    ↑ 新規追加        ↑ 高品質のみ                        ↑ 新規追加
```

---

## 実験計画

### Phase 1: 改善版の検証（短時間テスト）
```bash
python3 main_v2.py --n_iter 2 --num_epochs 2 --dir ./experiments/20251214/improved_v2_test
```
- 目的: 改善版が正しく動作するか確認
- 所要時間: 約10-15分

### Phase 2: 本格実験（改善版）
```bash
python3 main_v2.py --n_iter 5 --num_epochs 100 --dir ./experiments/20251214/improved_v2_full
```
- 目的: 改善効果を定量的に評価
- 所要時間: 約2-4時間（Early Stoppingにより短縮される可能性）

### Phase 3: 比較分析
- オリジナル版 vs 改善版v2の結果を比較
- 改善効果を可視化（グラフ、表）
- 最終レポートの作成

---

## 技術的な詳細

### トリプルスコアリングの実装

埋込モデル（TransE）によるスコアリング:
```
score(h, r, t) = -||h + r - t||
```

正規化（sigmoid）:
```
normalized_score = 1 / (1 + exp(-score))
```

パーセンタイルフィルタリング:
```
threshold = percentile(scores, min_score_percentile)
filtered = [triple for triple, score in zip(triples, scores) if score >= threshold]
```

### Early Stoppingの実装

```python
# 履歴: [s0, s1, s2, s3, ...]
# patience = 2の場合、s1 < s0 かつ s2 < s1 なら停止

is_degrading = all(recent[i] < recent[i-1] for i in range(1, len(recent)))
```

---

## リスクと制限事項

### 潜在的なリスク

1. **過度なフィルタリング**
   - `min_score_percentile`が高すぎると、有用なトリプルも除外される可能性
   - 推奨: 30-50の範囲で実験

2. **Early Stoppingの誤判定**
   - 短期的な悪化が長期的には改善につながる場合もある
   - 推奨: `patience`を2-3に設定

3. **計算コストの増加**
   - 品質フィルタリングにより、各iteration当たりの計算時間が10-20%増加
   - ただし、Early Stoppingにより全体としては短縮される

### 対処法

- 複数のパラメータ設定で実験を実施
- 結果を慎重に比較分析
- 必要に応じてパラメータを調整

---

## 次のステップ

1. **Phase 1実験**: 改善版の短時間テスト（✓ 次に実行）
2. **Phase 2実験**: 改善版の本格実験
3. **比較分析**: オリジナル vs 改善版
4. **最終レポート**: 改善効果の定量的評価

---

## 参考情報

### 関連ファイル

- **オリジナル版**: `/app/main.py`
- **改善版v2**: `/app/main_v2.py`
- **評価モジュール**: `/app/simple_active_refine/evaluation.py`
- **埋込モジュール**: `/app/simple_active_refine/embedding.py`

### テスト結果

- **テストディレクトリ**: `/app/experiments/20251214/test_eval/`
- **最終レポート**: `/app/experiments/20251214/test_eval/final_evaluation_report.md`
- **評価グラフ**: `/app/experiments/20251214/test_eval/evaluation_summary.png`

---

## 改善の根拠

### 機械学習の best practices

1. **Data Quality Matters**: 低品質データは高品質データに勝る量では補えない
2. **Early Stopping**: 過学習防止の標準的手法
3. **Validation-Driven**: 評価指標に基づく動的な制御

### 知識グラフ埋込の特性

- 埋込空間での距離が小さい = 尤もらしいトリプル
- ノイズの混入 → 埋込品質の劣化
- 高品質トリプルの追加 → 埋込品質の向上

---

## まとめ

改善版v2では、**品質重視のアプローチ**を採用しました。量より質を重視することで、対象トリプルのスコア改善とKG全体の精度向上の両立を目指します。

**Key Changes**:
1. ✅ トリプル品質フィルタリング（40パーセンタイル）
2. ✅ Early Stopping（patience=2）
3. ✅ 詳細なログ出力

これらの改善により、実験の効率化と結果の改善が期待されます。
