# REC-20260119-DROP_RATIO_1_ADD_ALL_KGFIT-001: drop_ratio=1.0 vs train_removed全投入（add-all）をKG-FITで再評価

作成日: 2026-01-19
最終更新日: 2026-02-01

## 目的
`drop_ratio=1.0`（reduced train）と `train ∪ train_removed`（add-all）を**KG-FIT KGE**で比較し、
**target_triples のスコア差（before vs after）**を再検証する。

## 実験セットアップ
- 対象データ: `/app/data/FB15k-237`
- 対象 relation: `/people/person/nationality`
- データ生成: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_20260119025151/dataset_drop1`
- add-all データ: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_20260119025151/retrain_eval/updated_triples`
- 埋込設定: `/app/config_embeddings_kgfit_fb15k237.json`
- 学習 epoch: 20
- 評価: minmax 正規化スコア（before/after の共通スコア可能 target_triples のみ）
- KG-FIT 近傍K: m=5（論文設定に合わせた default）

実行物:
- 実験ディレクトリ: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_kgfit_20260119210713/`
- サマリ: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_kgfit_20260119210713/retrain_eval/summary.json`

## データ分割の規模
`summary.json` より:
- base train in: 272,115
- removed: 129,230
- reduced train out（drop_ratio=1.0）: 142,885
- add-all 後 train: 272,115

## 結果
`summary.json` より（minmax 正規化スコア、共通29本）:
- before mean: 0.5686
- after mean: 0.2826
- mean(before - after): +0.2860

統計検定（参考値。direction を明示）:
- $H_1$: before < after
  - paired t-test p = 1.0
  - Wilcoxon p = 1.0
- $H_1$: before > after（観測された方向）
  - paired t-test p = 3.956e-32
  - Wilcoxon p = 1.863e-09
  - 効果量（paired Cohen’s d）= 11.81

## 結論（観測→含意→次アクション）

### 観測

- 共通スコア可能だった target_triples（29本）では、add-all 後の minmax 正規化スコアが大きく低下した（before mean 0.5686 → after mean 0.2826）。
- 統計検定でも、観測された方向（before > after）が強く支持された（paired t-test / Wilcoxon ともに p が極小）。

### 含意

- 本条件（KG-FIT, epoch=20, neighbor_k=5）では、add-all による target_triples の悪化が抑制されなかった。
- target_triples の多くが before/after のどちらかでスコア不能となっており、評価が 29 本に限定される点が解釈上の制約となる。

### 次アクション

- 修正版データセット（target_triples を train に含める）条件で、KG-FIT の drop_ratio=1.0 vs add-all を再評価する（例: REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-003）。
- 近傍K・正則化重みなど KG-FIT 側の感度（neighbor_k / regularization）をスイープし、add-all に対する頑健性が出る領域があるか確認する。

## 注意点 / 既知の制約
- target_triples 116本中、before/after両方でスコア可能だったのは 29本のみ。
- 評価（Hits/MRR）は省略（`skip_evaluation` 相当）し、target_triples のスコア比較に集中。

## 関連
- 旧（TransE）実験記録: [docs/records/REC-20260119-DROP_RATIO_1_ADD_ALL-001.md](REC-20260119-DROP_RATIO_1_ADD_ALL-001.md)
- dataset生成仕様: [docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md](../rules/RULE-20260119-MAKE_TEST_DATASET-001.md)
