# REC-20260119-DROP_RATIO_1_ADD_ALL-001: drop_ratio=1.0（文脈全削除） vs train_removed全投入（add-all）でのtarget_triplesスコア有意差検証

作成日: 2026-01-19
最終更新日: 2026-02-01

## 目的
`drop_ratio=1.0`（target head周辺の「非target relation」トリプルを最大限落とした reduced train）で学習したKGE（before）と、`train ∪ train_removed`（add-all）で学習したKGE（after）を比較し、

- **「drop_ratio=1.0 の方が target_triples のスコアが有意に低いか」**

を検証する。

## 実験セットアップ
- 対象データ: `/app/data/FB15k-237`
- 対象 relation: `/people/person/nationality`
- データ生成: `make_test_dataset.py`（設定: `/app/config_dataset.json`）
- 埋込学習設定: `/app/config_embeddings_with_stopper.json`
  - `valid.txt` が無いため early stopper は自動無効化（既存仕様）
- 学習 epoch: 20
- 比較:
  - before: `dataset_drop1/train.txt`（reduced train）
  - after: `updated_triples/train.txt`（reduced train に `train_removed` を全投入）

実行物:
- 実験ディレクトリ: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_20260119025151/`
- サマリ: `/app/experiments/20260119/exp_drop_ratio_1_vs_add_all_20260119025151/retrain_eval/summary.json`

## データ分割の規模
`summary.json` より:
- base train in: 272,115
- removed: 129,230
- reduced train out（drop_ratio=1.0）: 142,885
- add-all 後 train: 272,115

## 評価方法（統計検定）
- `target_triples.txt`（116本）を、**before/after両モデルで entity/relation が既知のもの**に限定してスコアリング
  - 実際にスコア可能だった共通集合: 29本
- スコア: `KnowledgeGraphEmbedding.score_triples(..., normalize=True, norm_method="minmax")`
- paired 検定:
  - paired t-test
  - Wilcoxon signed-rank test

検定の方向（ユーザ仮説）:
- $H_1$: before（drop_ratio=1.0 reduced） < after（add-all）

## 結果
`summary.json` より（minmax 正規化スコア、共通29本）:
- before mean: 0.5966
- after mean: 0.5564
- mean(before - after): +0.0402

統計検定（参考値。direction を明示）:
- $H_1$: before < after
  - paired t-test p = 0.9999999858
  - Wilcoxon p = 0.9999999534
- $H_1$: before > after（観測された方向）
  - paired t-test p = 1.4222e-08
  - Wilcoxon p = 6.1467e-08
  - 効果量（paired Cohen’s d）= 1.41

補足（IterationEvaluatorの raw target score。共通29本の平均）:
- target_score: -10.4207 -> -16.1979（Δ=-5.7773）
  - ※スコアの向きはモデル依存だが、minmaxでも after が低下しているため「targetに関しては悪化」と解釈してよい。

## 結論（観測→含意→次アクション）

### 観測

- 共通スコア可能だった target_triples（29本）では、add-all 後の minmax 正規化スコアが低下した（before mean 0.5966 → after mean 0.5564）。
- 統計検定でも、観測された方向（before > after）が強く支持された（paired t-test / Wilcoxon ともに p が極小）。

### 含意

- 当初の仮説「drop_ratio=1.0 の方が target_triples スコアが有意に低い」は支持されず、少なくとも本条件（TransE, epoch=20）では add-all が target に不利に働く可能性がある。
- ただし、評価は before/after 共通でスコア可能だった 29 本に限定されており、結論の適用範囲は限定的である。

### 次アクション

- `include_target`（target triple/target entities を train 側に強制的に含める）を明示して、target_triples のスコア可能率を上げた条件で同様の比較を再実行する。
- add-all による悪化の原因切り分けとして、追加トリプル中の predicate 分布や、target head 近傍の構造変化（degree/2-hop増分）を分析する。

## 注意点 / 既知の制約
- target_triples 116本中、before/after両方でスコア可能だったのは 29本のみ。
  - 多くが「どちらかの学習trainに entity が出現しない」などで落ちている可能性がある。
  - 今回の結論は「共通スコア可能集合」に限定される。

## 関連
- `train_removed 全投入`対照（v3データ）: `docs/records/REC-20260118-ADD_ALL_TRAIN_REMOVED-001.md`
- dataset生成の仕様: `docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md`
