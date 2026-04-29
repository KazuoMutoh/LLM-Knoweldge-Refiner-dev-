# REC-20260117-FULL_PIPELINE_PROGRESS-001: Full Pipeline Experiment Progress (2026-01-17)

作成日: 2026-01-17
最終更新日: 2026-01-17
更新日: 2026-01-17（5分監視とstopper修正の追記）

## 目的

`run_full_arm_pipeline.py` を用いた本格実験（計画: [docs/records/REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md](REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md)）の進捗と、運用上の不具合修正を記録する。

参照ルール:
- [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](../rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)

## 実行環境・入出力

- Dataset: `/app/experiments/test_data_for_nationality_v3`
- Target relation: `/people/person/nationality`
- Before model: `/app/models/20260116/fb15k237_transe_nationality`
- Run dir (baseline): `/app/experiments/20260117/exp_A_baseline`

## 進捗サマリ

### 実験 A（baseline）

- 状態: 完了
- 出力: `/app/experiments/20260117/exp_A_baseline/arm_run/retrain_eval/summary.json`

主要結果（summary.json より）:
- 反復: 10 iterations
- 追加 evidence: 530 unique triples
- train triples: 177,889 → 178,419 (+530)
- KGE metrics（before → after）:
  - Hits@1: 0.2426 → 0.2500 (Δ +0.00735)
  - Hits@3: 0.5392 → 0.4951 (Δ -0.0441)
  - Hits@10: 0.8235 → 0.7770 (Δ -0.0466)
  - MRR: 0.4257 → 0.4109 (Δ -0.0148)

※ baseline は「成立確認 + 小規模の挙動観察」を目的としていたため、改善が出ない（むしろ悪化する）こと自体は異常とはみなさない。Series B/C で反復数や selector, reward 重みの影響を比較する。

### 実験 B/C

- 状態: 完了（A の後に Series B/C を実行し、`summary.json` まで生成済み）
- 最終集計: [docs/records/REC-20260117-FULL_PIPELINE_RESULTS-001.md](REC-20260117-FULL_PIPELINE_RESULTS-001.md)

## 運用上の修正（再実行耐性）

- `run_all_experiments.sh` の再実行時に、既に完了している実験をスキップできるようにした。
  - 完了判定: `<run_dir>/arm_run/retrain_eval/summary.json` の存在
- `run_all_experiments.sh` が `run.log` を上書きしないよう、start 時のログ出力を append に変更した。

該当ファイル:
- `/app/experiments/20260116/run_all_experiments.sh`

## 監視（5分おき）

- 5分おきに runner の生存/進捗/失敗兆候をスナップショットする監視スクリプトを追加。
  - script: `/app/experiments/20260117/monitor_progress.sh`
  - log: `/app/experiments/20260117/_progress_watch.log`
  - pid: `/app/experiments/20260117/monitor.pid`

追記（2026-01-17）:
- 実験完了後、監視プロセスは停止し `monitor.pid` は削除した。

## 追加の不具合修正（early stopper + valid無し）

- `config_embeddings_with_stopper.json` 利用時、`valid.txt` が空のデータで PyKEEN early stopping が例外停止する問題に対応。
  - `simple_active_refine/embedding.py` で validation が無い場合は stopper を自動無効化。
  - 記録: `docs/records/REC-20260117-EARLY_STOPPER_VALIDATION_FIX-001.md`

## 次アクション

- Series B/C を `/app/experiments/20260117/` 配下に順次実行し、各実験で `summary.json` 生成までを監視する。
- 失敗時は `run.log` / `retrain_eval/` 出力を根拠に最小修正を行い、記録を追記する。
