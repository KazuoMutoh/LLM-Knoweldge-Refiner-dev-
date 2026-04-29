# RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001: retrain_and_evaluate_after_arm_run.py 実装仕様（arm-run後の再学習/評価）

作成日: 2026-01-17
最終更新日: 2026-02-01
更新日: 2026-01-17（early stopper + valid無しクラッシュ回避を追記）
更新日: 2026-01-17（accepted_added_triples.tsv を優先集約し、added_triples.tsv を出力するよう更新）
更新日: 2026-02-01（ep=2 の sanity → ep=100 の本番、cumulative union（iter全結合）での評価運用を追記）

---

## 0. 目的とスコープ

`retrain_and_evaluate_after_arm_run.py` は、arm-run（反復精錬）の出力から **accepted evidence** を集約し、
更新後データセット（train ∪ evidence）を作成したうえで、KGEの **before/after** を評価・比較し、
`summary.json` を含む成果物を `run_dir/retrain_eval/` 以下に保存する。

本仕様は、現行実装（2026-01-17時点）の **関数API**と **CLI**、成果物構造、モデルディレクトリの前提、例外/終了条件を再実装可能な粒度で定義する。

---

## 1. 提供API

### 1.1 関数: `run(...) -> Path`

入力（概念）:
- `run_dir: str|Path`
  - arm-run出力ディレクトリ（`iter_*/accepted_evidence_triples.tsv` を含む）
- `dataset_dir: str|Path`
  - 元データセット（`train.txt` を必須。`valid.txt`/`test.txt` は存在すればコピー）
- `target_triples: str|Path`
  - ターゲットトリプルTSV（score比較対象）
- `model_before_dir: str|Path|None`
  - default: `<run_dir>/retrain_eval/model_before`
- `model_after_dir: str|Path|None`
  - default: `<run_dir>/retrain_eval/model_after`
- `exclude_predicate: Sequence[str]|None`
  - evidence追加から除外するpredicate（repeatable想定）
- `after_mode: str`（`"load"` / `"retrain"`）
- `embedding_config: str|Path`
  - after_mode=retrain のときの学習設定JSON
- `num_epochs: int`
- `force_retrain: bool`

出力（戻り値）:
- `out_base: Path`（`<run_dir>/retrain_eval`）

---

## 2. 入力仕様

### 2.1 arm-run出力ディレクトリ `run_dir`

集約対象ファイル:
- 優先: `iter_*/accepted_added_triples.tsv`（evidence + incident candidates の和集合）
- 互換: `iter_*/accepted_evidence_triples.tsv`（旧形式。accepted_added が無い場合のfallback）

運用メモ（cumulative union）:
- 本スクリプトの集約は、`run_dir` 配下の `iter_*/accepted_added_triples.tsv` を全て読み、ユニーク集合を作る。
- そのため、基本は「iter=1..final の累積 union」での評価になる（incremental を評価したい場合は `run_dir` を分けるか、対象iterだけを残して評価する）。

TSV形式:
- 3列: `head<TAB>relation<TAB>tail`

### 2.2 学習済みモデルディレクトリ（before/after共通）

`KnowledgeGraphEmbedding` がロードできる “保存済みPyKEEN run” 形式であること。

必須:
- `trained_model.pkl`
- `training_triples/`

検証:
- `_validate_trained_model_dir(model_dir, label)` が上記を満たさない場合 `SystemExit(2)` で停止。

重要（運用上の注意）:
- デフォルト `model_before_dir` は `run_dir/retrain_eval/model_before`。
- 本スクリプトは beforeモデルを自動コピーしない。
  - 事前に `--model_before_dir` を正しく指定するか、デフォルトパスにモデルを配置する必要がある。

### 2.3 `embedding_config`（after_mode=retrain）

JSON object（dict）であること。

期待キー（概念）:
- `model`, `model_kwargs`, `loss`, `training_loop`, `optimizer`, `optimizer_kwargs`, `training_kwargs`, `random_seed`, ...
- `embedding_backend`: `"pykeen" | "kgfit"`（省略時 `pykeen`）
- `kgfit`: `embedding_backend="kgfit"` の場合の追加設定（省略可）
- `training_kwargs.num_epochs` はこのスクリプトが `num_epochs` で上書きする

注意:
- 互換性のため `cfg.pop("num_epochs", None)` でトップレベルの `num_epochs` は削除される。

#### KG-FITバックエンド使用時の前提

`embedding_backend="kgfit"` を指定した場合、学習は次を前提とする。
- `dataset_dir`（= `dir_triples`）に `.cache/kgfit/` の事前計算成果物が存在する
  - `entity_name_embeddings.npy` / `entity_desc_embeddings.npy` / `entity_embedding_meta.json`
  - `hierarchy_seed.json` / `cluster_embeddings.npy` / `neighbor_clusters.json`
- 現状は `model="TransE"` のみサポート

詳細仕様:
- KG-FITバックエンドの標準: [docs/rules/RULE-20260119-TAKG_KGFIT-001.md](RULE-20260119-TAKG_KGFIT-001.md)

#### early stopping（stopper）に関する注意

- `stopper="early"`（PyKEENのearly stopping）は **validation triples が存在することが前提**。
- `valid.txt` が空（または存在しない）場合に `stopper="early"` を有効のまま渡すと、PyKEEN側で例外により学習が停止する場合がある。
- 現行実装では、`simple_active_refine/embedding.py` の `KnowledgeGraphEmbedding.train_model()` が
  `stopper` が設定されているのに `valid_tf is None` の場合、warning を出して **自動的に stopper を無効化**して学習を継続する。

---

## 3. 出力仕様（固定ディレクトリ構造）

`out_base = <run_dir>/retrain_eval` 配下:

- `updated_triples/`
  - `train.txt`（元train ∪ evidence（除外適用後））
  - `valid.txt`, `test.txt`（元datasetからコピー。存在する場合）
  - `added_triples.tsv`（集約した追加トリプル。exclude適用前の「集約結果」を保存）
  - `added_evidence.tsv`（後方互換用。現行は `added_triples.tsv` と同内容を保存）
- `model_before/`（デフォルトパス。実体はユーザが用意）
- `model_after/`（after_mode=retrainならここに新規学習結果を保存）
- `evaluation/`（`IterationEvaluator.evaluate_iteration(..., dir_save=...)` の出力）
- `summary.json`（本スクリプトの最終サマリ）

---

## 4. 処理フロー

1. `out_base` と `updated_triples_dir` を決定し、`out_base.mkdir(..., exist_ok=True)`
2. evidence集約
  - `aggregate_accepted_added_triples(run_dir)`
  - 結果（unique triples）を `updated_triples/added_triples.tsv` に保存（互換のため `added_evidence.tsv` も保存）
3. 更新後データセット作成
   - `create_updated_triples_dir(dataset_dir, out_dir=updated_triples_dir, evidence_triples, exclude_predicates=exclude_predicate)`
4. beforeモデル検証・評価用split補完
   - `_validate_trained_model_dir(model_before_dir, "before")`
   - `_ensure_model_dir_has_splits(source_triples_dir=dataset_dir, model_dir=model_before_dir)`
5. target triples 読み込み
   - `read_triples(target_triples_path)`
6. KGEロード/作成
   - before: `KnowledgeGraphEmbedding(model_before_dir)`
   - after:
     - `after_mode=="load"`: `_validate_trained_model_dir(model_after_dir, "after")` → `KnowledgeGraphEmbedding(model_after_dir)`
     - `after_mode=="retrain"`: `_train_after_kge(...)` を実行
       - 非空 `model_after_dir` の上書きは `force_retrain=True` の場合のみ許可（違反は `SystemExit(2)`）
       - `updated_triples_dir` から `test.txt`/`valid.txt` を `model_after_dir` にコピー（評価要件）
       - `embedding_config` を読み、`training_kwargs.num_epochs=num_epochs` を強制
       - `KnowledgeGraphEmbedding.train_model(**cfg)`
7. 評価
   - `IterationEvaluator().evaluate_iteration(iteration=1, kge_before, kge_after, target_triples, n_triples_added, dir_save=eval_dir)`
8. `summary.json` 保存

---

## 5. 失敗時の挙動

- before/after（load時）のモデルディレクトリが不正: `SystemExit(2)`
- after_mode=retrain で `model_after_dir` が非空かつ `force_retrain=False`: `SystemExit(2)`
- `embedding_config` が dict でない: `ValueError`
- `after_mode` が未知: `ValueError`
- その他I/O失敗: 例外伝播

---

## 6. CLI仕様

コマンド:
- `python retrain_and_evaluate_after_arm_run.py ...`

必須:
- `--run_dir`
- `--dataset_dir`
- `--target_triples`

主なオプション:
- `--model_before_dir`（default: `<run_dir>/retrain_eval/model_before`）
- `--model_after_dir`（default: `<run_dir>/retrain_eval/model_after`）
- `--exclude_predicate`（repeatable）
- `--after_mode {load,retrain}`（default: `retrain`）
- `--embedding_config`（default: `./config_embeddings.json`）
- `--num_epochs`（default: 2）
- `--force_retrain`（after_mode=retrain の上書き許可）

推奨運用（再現性/コスト/失敗切り分け）:
- まず `--num_epochs 2` で sanity を通し、I/O（updated_triples/ と summary.json）が揃うことを確認する
- 次に `--num_epochs 100` で本番評価を実行する

---

## 7. 実装参照（ソースコード）

- 実装: [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)
- 集約/更新: [simple_active_refine/dataset_update.py](../../simple_active_refine/dataset_update.py)
- KGE: [simple_active_refine/embedding.py](../../simple_active_refine/embedding.py)
- 評価: [simple_active_refine/evaluation.py](../../simple_active_refine/evaluation.py)

---

## 8. 参考文献（docs）

- 統合ランナー仕様（このステップを呼ぶ）: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- arm-run後評価の実装計画: [docs/records/REC-20260114-KGE_RETRAIN-001.md](../records/REC-20260114-KGE_RETRAIN-001.md)
- arm-run後評価の初期計画（beforeモデル必須など）: [docs/records/REC-20260111-ARM_REFINE-007.md](../records/REC-20260111-ARM_REFINE-007.md)
- arm設計（after_mode, 出力規約）: [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
