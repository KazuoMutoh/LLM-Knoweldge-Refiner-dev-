# REC-20260120-DROP_RATIO_1_ADD_ALL-002: drop_ratio=1.0 vs add-all（TransE, epoch=100）

作成日: 2026-01-20
最終更新日: 2026-02-01

## 目的
`drop_ratio=1.0`（reduced train）と `train ∪ train_removed`（add-all）を**TransE KGE**で比較し、
**epoch=100** で target_triples のスコア差を再検証する。

## 実験セットアップ
- 対象データ: `/app/data/FB15k-237`
- 対象 relation: `/people/person/nationality`
- データ生成: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_20260119025151/dataset_drop1`
- add-all データ: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_20260119025151/retrain_eval/updated_triples`
- 埋込設定: `/app/config_embeddings_with_stopper.json`
- 学習 epoch: 100
- 評価: minmax 正規化スコア（before/after の共通スコア可能 target_triples のみ）

実行物:
- 実験ディレクトリ: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_transe_ep100_20260120025524/`
- サマリ: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_transe_ep100_20260120025524/retrain_eval/summary.json`

## データ分割の規模
`summary.json` より:
- base train in: 272,115
- removed: 129,230
- reduced train out（drop_ratio=1.0）: 142,885
- add-all 後 train: 272,115

## 結果
`summary.json` より（minmax 正規化スコア、共通29本）:
- before mean: 0.6192
- after mean: 0.8614
- mean(before - after): -0.2422

統計検定（参考値。direction を明示）:
- $H_1$: before < after（観測された方向）
  - paired t-test p = 6.511e-31
  - Wilcoxon p = 1.863e-09
- $H_1$: before > after
  - paired t-test p = 1.0
  - Wilcoxon p = 1.0
  - 効果量（paired Cohen’s d）= -10.68

## 結論（観測→含意→次アクション）

### 観測

- 共通スコア可能だった target_triples（29本）では、add-all 後の minmax 正規化スコアが有意に上昇した（before mean 0.6192 → after mean 0.8614）。

### 含意

- epoch=20（REC-20260119-DROP_RATIO_1_ADD_ALL-001）とは逆方向の結果であり、学習の収束度合い（epoch）により before/after の優劣が反転し得る。
- 本条件も共通スコア可能集合が 29 本に限定されるため、target_triples 全体に一般化するには制約がある。

### 次アクション

- 修正版データセット（target_triples を train に含める）で、全116本を対象に同様の比較を確認する（REC-20260120-DROP_RATIO_1_ADD_ALL-003）。
- KG-FIT 条件（REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-002）と揃えた設定で再比較し、埋め込み方式の差を切り分ける。

## 注意点 / 既知の制約
- target_triples 116本中、before/after両方でスコア可能だったのは 29本のみ。
- 評価（Hits/MRR）は省略（`skip_evaluation` 相当）し、target_triples のスコア比較に集中。

## 関連
- 旧（TransE, epoch=20）: [docs/records/REC-20260119-DROP_RATIO_1_ADD_ALL-001.md](REC-20260119-DROP_RATIO_1_ADD_ALL-001.md)
- KG-FIT比較（epoch=100）: [docs/records/REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-002.md](REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-002.md)
- dataset生成仕様: [docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md](../rules/RULE-20260119-MAKE_TEST_DATASET-001.md)
