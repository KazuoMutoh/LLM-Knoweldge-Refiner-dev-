# 改善報告書（修正版）

## 実施日: 2024年12月14日

## 背景
テスト実験（epoch=2, n_iter=2）の結果、対象トリプルのスコアが-73%悪化する問題が発生。
原因と改善策を再検討した結果、技術的な制約を考慮した正しいアプローチを特定。

## 技術的制約の認識

### 重要な制約
**候補トリプルのスコアリングは不可能**
- 候補トリプルに含まれるエンティティは知識グラフに未登録（新規エンティティ）
- 埋込モデルは学習済みエンティティのみスコアリング可能
- したがって、候補トリプルを事前にフィルタリングすることはできない

### 正しいアプローチ
高品質なルールから抽出されたトリプルを**そのまま使用**

## 改善策（修正版）

### 1. 高品質ルール抽出の導入

#### 問題点
- 現在の実装は知識グラフ全体からルール抽出
- 低品質なパターンも含まれる可能性

#### 改善策
**スコアの高いトリプルのk-hop近傍からルール抽出**

```python
# 既存実装を活用
amie_rules = extract_rules_from_high_score_triples(
    kge=kge_initial,
    target_relation=config_dataset['target_relation'],
    lower_percentile=80.0,  # 上位20%のトリプル
    k_neighbor=1,           # 1-hop近傍
    min_head_coverage=0.01,
    min_pca_conf=0.05
)
```

**メリット**:
- 実際にスコアの高いトリプル周辺の頻出パターンを抽出
- より有望なルールを発見できる可能性
- `simple_active_refine/rule_extractor.py`に既に実装済み

### 2. 複合的なルール選択基準

#### 問題点
- 現在はPCA confidenceのみで選択
- 偏ったルールプールになる可能性

#### 改善策
**複数の指標を組み合わせたルール選択**

```python
def select_diverse_rules(
    amie_rules,
    n_rules: int,
    pca_weight: float = 0.4,      # PCA confidence
    coverage_weight: float = 0.3,  # Head coverage
    diversity_weight: float = 0.3  # Body simplicity
):
    # 各指標を正規化
    pca_norm = normalize(pca_scores)
    coverage_norm = normalize(coverage_scores)
    diversity_norm = 1.0 / (body_sizes + 1.0)  # シンプルなルールを優先
    
    # 複合スコア計算
    composite_scores = (
        pca_weight * pca_norm +
        coverage_weight * coverage_norm +
        diversity_weight * diversity_norm
    )
    
    # 上位n_rules個を選択
    return select_top_k(composite_scores, n_rules)
```

**考慮する指標**:
- **PCA Confidence**: ルールの正確性
- **Head Coverage**: ルールの適用範囲
- **Body Size**: ルールの単純性（小さいほど良い）

**メリット**:
- バランスの取れたルールプール
- 多様なパターンをカバー
- より堅牢な改善

### 3. Early Stopping機能の追加

#### 問題点
- スコアが悪化し続けても処理を継続
- 無駄な計算リソースの消費

#### 改善策
**連続悪化を検出して停止**

```python
def check_iteration_quality(
    target_score_history: List[float],
    patience: int = 2
) -> bool:
    if len(target_score_history) < patience + 1:
        return True
    
    # 直近patience回が全て悪化しているかチェック
    recent = target_score_history[-(patience+1):]
    is_degrading = all(recent[i] < recent[i-1] for i in range(1, len(recent)))
    
    if is_degrading:
        logger.warning('Target score has degraded for consecutive iterations')
        return False
    
    return True
```

**メリット**:
- 無駄な計算を削減
- 早期に問題を検出
- より効率的な実験

## 改善版の実装

### ファイル
- `/app/main_v2_revised.py`: 改善版メインスクリプト

### 主要な変更点

1. **コマンドライン引数追加**:
```bash
--use_high_score_triples   # 高スコアトリプルからルール抽出
--lower_percentile 80.0    # スコア閾値（パーセンタイル）
--k_neighbor 1             # k-hop近傍
--early_stop_patience 2    # Early Stopping許容回数
```

2. **ルール抽出方法の切り替え**:
```python
if use_high_score_triples:
    # 【改善】高スコアトリプルのk-hop近傍からルール抽出
    amie_rules = extract_rules_from_high_score_triples(...)
else:
    # オリジナル：知識グラフ全体からルール抽出
    amie_rules = extract_rules_from_entire_graph(...)
```

3. **複合基準でのルール選択**:
```python
rule_pool = select_diverse_rules(
    amie_rules=amie_rules,
    n_rules=n_rules_pool,
    pca_weight=0.4,
    coverage_weight=0.3,
    diversity_weight=0.3
)
```

4. **Early Stopping**:
```python
target_score_history.append(iteration_metrics.target_score_after)

if not check_iteration_quality(target_score_history, patience=early_stop_patience):
    logger.warning('EARLY STOPPING: Target score degradation detected')
    break
```

## 実験計画

### Phase 1: 改善版のテスト
```bash
python main_v2_revised.py \
  --n_iter 2 \
  --num_epochs 2 \
  --dir ./experiments/20251214/improved_v2_test \
  --use_high_score_triples \
  --lower_percentile 80.0 \
  --k_neighbor 1 \
  --early_stop_patience 2
```

### Phase 2: 本格実験（改善版）
```bash
python main_v2_revised.py \
  --n_iter 3 \
  --num_epochs 100 \
  --dir ./experiments/20251214/improved_v2 \
  --use_high_score_triples \
  --lower_percentile 80.0 \
  --k_neighbor 1 \
  --early_stop_patience 2
```

### Phase 3: オリジナル版との比較
```bash
# オリジナル版（既に実行中）
./experiments/20251214/original

# 改善版
./experiments/20251214/improved_v2
```

## 期待される効果

### 1. 高品質ルール抽出
- スコアの高いトリプル周辺のパターン → より有望なルール
- 実データに基づく頻出パターン → 高い適用可能性

### 2. 複合的な選択基準
- PCA + Coverage + Simplicity → バランスの取れたルールプール
- 多様なパターン → 広範なケースをカバー

### 3. Early Stopping
- 連続悪化の検出 → 無駄な計算削減
- より効率的な実験 → リソースの有効活用

## 評価指標

### 定量評価
1. **対象トリプルスコア変化**: 改善されているか
2. **Hits@k**: リンク予測精度の向上
3. **MRR**: ランキング品質の向上
4. **追加トリプル数**: 効率的な知識追加

### 定性評価
1. **ルール品質**: より有望なパターンを抽出できているか
2. **多様性**: 偏りのないルールプールになっているか
3. **安定性**: Early Stoppingが適切に機能するか

## タイムライン

| 時刻 | タスク | 状態 |
|------|--------|------|
| 09:50 | オリジナル版実験開始 | 実行中（epoch 32/100） |
| 10:30-11:00 | オリジナル版実験完了予定 | 待機中 |
| 11:00-11:15 | 結果分析 | 待機中 |
| 11:15-11:20 | 改善版テスト実験 | 待機中 |
| 11:20-13:00 | 改善版本格実験 | 待機中 |
| 13:00- | 比較分析・レポート作成 | 待機中 |

## リスクと対策

### リスク1: 高スコアトリプルからのルール抽出が失敗
- **対策**: lower_percentileを調整（90% → 80% → 70%）
- **代替案**: オリジナル版（entire_graph）を使用

### リスク2: 複合基準での選択が効果的でない
- **対策**: ウェイトを調整（pca_weight, coverage_weight, diversity_weight）
- **代替案**: 単一基準での選択も試行

### リスク3: Early Stoppingが早すぎる
- **対策**: patience値を増やす（2 → 3 → 4）
- **代替案**: Early Stoppingを無効化

## まとめ

### 主な改善点
1. ✅ 高品質ルール抽出（既存実装活用）
2. ✅ 複合的な選択基準（新規実装）
3. ✅ Early Stopping（新規実装）

### 技術的正しさ
- ❌ 候補トリプルのスコアリング → **不可能**（新規エンティティ含む）
- ✅ 高スコアトリプルからのルール抽出 → **実装済み**（`extract_rules_from_high_score_triples`）
- ✅ 複合基準での選択 → **新規実装**（`select_diverse_rules`）

### 次のステップ
1. オリジナル版実験の完了待ち
2. 結果の詳細分析
3. 改善版のテスト実験
4. 改善版の本格実験
5. 比較分析とレポート作成
