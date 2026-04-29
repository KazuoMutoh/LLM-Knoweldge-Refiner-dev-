# REC-20260117-FULL_PIPELINE_INTERIM_ANALYSIS-001: Full Pipeline 実験 途中経過分析（A / B1 / B2）

作成日: 2026-01-17
最終更新日: 2026-01-17

## 目的

`run_full_arm_pipeline.py` を用いた本格実験（計画: [docs/records/REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md](REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md)）について、完了済み実験（A / B1 / B2）の `summary.json` を集計し、途中経過を定量的に把握する。

追記（2026-01-17）:
- Series B/C も完了し、全 `summary.json` を集計した最終結果を [docs/records/REC-20260117-FULL_PIPELINE_RESULTS-001.md](REC-20260117-FULL_PIPELINE_RESULTS-001.md) にまとめた。

参照:
- 仕様: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](../rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- retrain/eval: [docs/rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md](../rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md)

## 対象（完了済み）

- A: `/app/experiments/20260117/exp_A_baseline/arm_run/retrain_eval/summary.json`
- B1: `/app/experiments/20260117/exp_B1_ucb_25/arm_run/retrain_eval/summary.json`
- B2: `/app/experiments/20260117/exp_B2_ucb_50/arm_run/retrain_eval/summary.json`

※ 本ドキュメントは途中経過（A/B1/B2）時点の記録であり、最終集計は別ドキュメントに移行した。

## 集計結果（summary.json のみ）

| exp | n_iter | epochs(after) | added evidence (unique) | train size before→after | target score Δ | Hits@1 Δ | Hits@3 Δ | Hits@10 Δ | MRR Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline | 10 | 10 | 530 | 177,889→178,419 | -0.2676 | +0.00735 | -0.0441 | -0.0466 | -0.0148 |
| B1_ucb_25 | 25 | 100 | 821 | 177,889→178,710 | -0.2565 | -0.0294 | -0.0123 | +0.00980 | -0.0179 |
| B2_ucb_50 | 50 | 100 | 892 | 177,889→178,781 | -0.4607 | +0.00245 | -0.0172 | +0.00000 | -0.000750 |

補足:
- `target_score` はモデルのスコア（大きい/小さいのどちらが良いかは実装依存）。この実験では before→after で負方向に動いている。
- Hits/MRR は Rank-based evaluation のため「大きいほど良い」。

## 所見（暫定）

1) 追加evidence数は `n_iter` を増やすほど単調増加ではない
- 10 iter: 530
- 25 iter: 821
- 50 iter: 892

50 iter で増分が鈍化しており、候補集合やルール選択の重複によって「新規 evidence が枯れてくる」兆候。

2) metrics は一貫して改善していない（ただし B2 は MRR 低下が小さい）
- A は Hits@3/10, MRR が下落。
- B1 は Hits@10 が僅かに上がる一方で Hits@1 と MRR が下落。
- B2 は Hits@10 が維持、MRR 低下が非常に小さい（-0.00075）。

現時点では「evidence量が増えるほど（必ずしも）KGE評価が良くなる」傾向は見えない。

3) retrain の epoch 数の影響が混ざっている
- A は epochs=10
- B1/B2 は epochs=100

A と B の差は `n_iter` だけでなく学習条件も異なるため、B3 以降の同条件比較（epochs=100, stopper-config）での傾向確認が重要。

## 次の分析アクション（提案）

- B3（ucb_100）完了後に、B1/B2/B3 を同条件として比較（evidence数の飽和と metrics の関係）。
- Series B の LLM-policy / Random 完了後に、同一 n_iter で selector 戦略差を比較（rule 探索の多様性が evidence の新規性に効くか）。
- 可能なら、各 `iter_k/accepted_evidence_triples.tsv` の「重複率」「predicate分布」を集計して、metrics 変動の原因（ノイズ混入 vs 有益な追加）を切り分け。
