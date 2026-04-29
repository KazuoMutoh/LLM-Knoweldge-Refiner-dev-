# REC-20260118-ADD_ALL_TRAIN_REMOVED-001: train_removed.txt 全投入で target score は上がるか？（2026-01-18）

作成日: 2026-01-18

参照:
- [REC-20260118-FULL_PIPELINE_RESULTS-001](REC-20260118-FULL_PIPELINE_RESULTS-001.md): 2026-01-18 の full pipeline 結果集約
- [RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001](../rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md): updated_triples 作成と before/after 評価の仕様（本実験も同等の形式で出力）

---

## 0. 背景と目的

「arm-run で追加したトリプルで target_triples のスコアが改善しない」現象について、そもそも候補集合である `train_removed.txt` を **全部** train に戻したとしても、KGE の target score が上がらない（むしろ下がる）可能性がある。

そこで、本記録では次を確認する：

- `train.txt ∪ train_removed.txt` で再学習しても target score が改善しない（または悪化する）か？
- その場合、これは「追加すれば必ず良くなる」という前提が成立していないことを意味する。

---

## 1. 実験設定

- dataset_dir: [experiments/test_data_for_nationality_v3](../../experiments/test_data_for_nationality_v3)
- target relation: `/people/person/nationality`
- target_triples: [target_triples.txt](../../experiments/test_data_for_nationality_v3/target_triples.txt)（117）
- add-all 対象: [train_removed.txt](../../experiments/test_data_for_nationality_v3/train_removed.txt)
- before model: [models/20260116/fb15k237_transe_nationality](../../models/20260116/fb15k237_transe_nationality)
- after model: `train.txt ∪ train_removed.txt` で再学習
- embedding config: [config_embeddings_with_stopper.json](../../config_embeddings_with_stopper.json)
- num_epochs: 100
  - valid が空のため early stopper は自動無効化（embedding wrapper の安全化挙動）

実行ディレクトリ:
- run_dir: [experiments/20260118/exp_add_all_train_removed_20260118a](../../experiments/20260118/exp_add_all_train_removed_20260118a)

再現コマンド:
- `cd /app && /app/.venv/bin/python /app/tmp/debug/run_add_all_train_removed_experiment.py --run-dir /app/experiments/20260118/exp_add_all_train_removed_20260118a --dataset-dir /app/experiments/test_data_for_nationality_v3 --train-removed /app/experiments/test_data_for_nationality_v3/train_removed.txt --target-triples /app/experiments/test_data_for_nationality_v3/target_triples.txt --model-before-dir /app/models/20260116/fb15k237_transe_nationality --embedding-config /app/config_embeddings_with_stopper.json --num-epochs 100`

---

## 2. 成果物

- summary: [summary.json](../../experiments/20260118/exp_add_all_train_removed_20260118a/retrain_eval/summary.json)
- log: [run.log](../../experiments/20260118/exp_add_all_train_removed_20260118a/run.log)
- updated dataset: [updated_triples](../../experiments/20260118/exp_add_all_train_removed_20260118a/retrain_eval/updated_triples)
- evaluation: [iteration_metrics.json](../../experiments/20260118/exp_add_all_train_removed_20260118a/retrain_eval/evaluation/iteration_metrics.json)

---

## 3. 結果

データセット規模:
- train_before: 177,889
- train_removed（追加に使用）: 94,226
- train_after: 272,115

KGE 指標（test set）と target score:
- Target score: -9.8677 → -15.9451（Δ=-6.0774）
- Hits@1: 0.2426 → 0.2304（Δ=-0.0123）
- Hits@3: 0.5392 → 0.4975（Δ=-0.0417）
- Hits@10: 0.8235 → 0.8015（Δ=-0.0221）
- MRR: 0.4257 → 0.4054（Δ=-0.0203）

備考:
- target_triples のうち unknown entity/relation により 11 件がスコア計算から除外される（before/after とも同様の挙動）。

---

## 4. 解釈（暫定）

- `train_removed.txt` を全投入しても target score は改善せず、むしろ大きく悪化した。
- したがって「候補集合を全部足せば target が上がる」という前提は、この設定（TransE + 本データ）では成立しない。

考えられる要因（仮説）:
- 追加トリプルが `/people/person/nationality` の説明に寄与しない（無関係/遠い）
- 多数の制約追加により、埋め込みが target relation を満たす配置から外れる
- ノイズ/矛盾の混入

---

## 5. 次アクション案

- `train_removed.txt` 全投入は上限ベースラインとしては有用だが、改善目的には不適。
- 改善のためには「どれを足すか（predicate/近傍/信頼度）」の選別が必要。
  - 例: target head（?a）近傍に限定、predicate をフィルタ、relation priors を使ったスコアリングで重み付け等。

---

## 更新履歴

- 2026-01-18: 新規作成。train_removed 全投入の再学習・評価を実施し、target score が悪化することを確認。
