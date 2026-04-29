# REC-20260120-DROP_RATIO_1_ADD_ALL-003: drop_ratio=1.0 vs add-all（TransE, epoch=100, target_triples=116）

作成日: 2026-01-20
最終更新日: 2026-02-01

## 目的
修正版 `make_test_dataset.py` を用い、target_triples を train に含めた状態で
`drop_ratio=1.0` vs `add-all` を**TransE**で再検証する。

## 実験セットアップ
- 対象データ: `/app/data/FB15k-237`
- 対象 relation: `/people/person/nationality`
- データ生成: `/app/experiments/20260120/exp_drop_ratio_1_vs_add_all_fixed_20260120_20260120114945/dataset_drop1`
- add-all データ: `/app/experiments/20260120/exp_drop_ratio_1_vs_add_all_fixed_20260120_20260120114945/retrain_eval/updated_triples`
- 埋込設定: `/app/config_embeddings.json`
- 学習 epoch: 100
- 評価: minmax 正規化スコア（before/after 共通スコア可能 target_triples のみ）

実行物:
- 実験ディレクトリ: `/app/experiments/20260120/exp_drop_ratio_1_vs_add_all_transe_ep100_fixed_20260120115345/`
- サマリ: `/app/experiments/20260120/exp_drop_ratio_1_vs_add_all_transe_ep100_fixed_20260120115345/retrain_eval/summary.json`

## データ分割の規模
`summary.json` より:
- base train in: 272,115
- removed: 129,143
- reduced train out（drop_ratio=1.0）: 142,972
- add-all 後 train: 272,115

## 結果
`summary.json` より（minmax 正規化スコア、共通116本）:
- before mean: 0.7165
- after mean: 0.8565
- mean(before - after): -0.13995

統計検定（参考値。direction を明示）:
- $H_1$: before < after（観測された方向）
  - paired t-test p = 1.407e-91
  - Wilcoxon p = 1.407e-91
- $H_1$: before > after
  - paired t-test p = 1.0
  - Wilcoxon p = 1.0

## 結論（観測→含意→次アクション）

### 観測

- 修正版データセット（target_triples を train に含める）では、target_triples 116本すべてが before/after 共通でスコア可能となった。
- その 116 本に対して、add-all 後の minmax 正規化スコアが有意に上昇した（before mean 0.7165 → after mean 0.8565）。

### 含意

- 共通スコア可能集合の小ささ（29本）が結論を不安定化させていた可能性が高く、target を train に含めることで評価の再現性が改善する。
- drop_ratio=1.0 vs add-all の比較は「target_triples/target entities が学習時に既知である」前提を満たすよう、データ生成側の仕様を明確化すべきである。

### 次アクション

- KG-FIT 条件でも同様に修正版データセットで再比較し、結論が一致するか確認する（REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-003）。
- target_triples のスコア差だけでなく、Hits/MRR 等のランキング指標や、追加トリプルの質（predicate 分布/近傍構造）も合わせて比較する。

## 注意点 / 既知の制約
- target_triples 116本中、before/after 両方でスコア可能だったのは 116本。
- 評価（Hits/MRR）は省略（`skip_evaluation` 相当）し、target_triples のスコア比較に集中。

## 関連
- 旧（TransE, epoch=100, target_triples=29）: [docs/records/REC-20260120-DROP_RATIO_1_ADD_ALL-002.md](REC-20260120-DROP_RATIO_1_ADD_ALL-002.md)
- KG-FIT比較（epoch=100, target_triples=116）: [docs/records/REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-003.md](REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-003.md)
- dataset生成仕様: [docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md](../rules/RULE-20260119-MAKE_TEST_DATASET-001.md)
