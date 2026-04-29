# REC-20260114-KGE_RETRAIN-001: arm-run後のKGE評価を「afterのみ再学習」対応に拡張する実装計画

作成日: 2026-01-14  
最終更新日: 2026-01-14

更新日: 2026-01-14
- テスト方針を更新: epoch=2で **実際にKGEを学習/評価**し、モック禁止（重いが正しさ優先）。
- 実装完了: after_mode=retrain を追加し、updated_triples で after を再学習するように変更。pytest は成果物保持＋ログ表示に変更。

## 0. 目的

現状の [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py) は
- arm-runの `accepted_evidence_triples.tsv` を集約して `updated_triples/` を作る
- before/after ともに **既存の学習済みモデルを読み込んで評価のみ**

となっている。

これを次の仕様に拡張する。
- **before KGE**: 再学習しない（学習済みを読み込む）
- **after KGE**: `updated_triples/` を入力に **再学習して作成**し、そのモデルで評価する

## 1. 現状の仕様（確認）

対象: [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)

- `aggregate_accepted_evidence_triples(run_dir)`
- `create_updated_triples_dir(dataset_dir, out_dir, evidence_triples, exclude_predicates)`
- before/after の `model_dir` に `trained_model.pkl` と `training_triples/` が無ければ終了
- `IterationEvaluator.evaluate_iteration()` で before/after の Hits@k/MRR と target triple score を比較

## 2. 追加したい仕様（要求仕様）

### 2.1 追加機能
- `updated_triples/` を入力に、`KnowledgeGraphEmbedding.train_model()` を用いて after KGE を学習
- 学習済み after モデルを `model_after_dir` に保存
- その after を用いて評価し、既存と同様に `retrain_eval/` 配下へ保存

### 2.2 既存互換（安全策）
実験運用上、「afterを再学習するモード」と「afterも既存モデルを使うモード」が両方欲しくなる可能性が高い。
そのため以下のどちらかのUXにする。

- 案A（推奨）: `--after_mode {load,retrain}` を追加し、デフォルトを `retrain` にする
- 案B: `--retrain_after`（bool, default True）を追加する

本計画では、UXが明確な案Aを採用する。

## 3. CLI 設計（追加・変更）

### 3.1 追加引数
- `--after_mode` : `load|retrain`（default: `retrain`）
- `--embedding_config` : after 学習用の embedding 設定 JSON（default: `./config_embeddings.json`）
- `--num_epochs` : after 学習の epoch 数（default: 設定ファイルの値があればそれ、無ければ 2 など最小）
- `--force_retrain` : `model_after_dir` が既に存在する場合に上書き実行を許可（default: False）

### 3.2 既存引数は維持
- `--run_dir`, `--dataset_dir`, `--target_triples`
- `--model_before_dir`, `--model_after_dir`
- `--exclude_predicate`

## 4. 実装方針（コード変更の骨子）

### 4.1 after 学習処理
- `updated_triples_dir` を作成済みであることを前提に、次を実行:
  - `embedding_config = json.load(embedding_config_path)`
  - `embedding_config["dir_triples"] = str(updated_triples_dir)`
  - `embedding_config["dir_save"] = str(model_after_dir)`
  - 必要なら `embedding_config["training_kwargs"]["num_epochs"] = args.num_epochs`
  - `kge_after = KnowledgeGraphEmbedding.train_model(**embedding_config)`

### 4.2 評価時の test/valid の取り扱い（重要）
`KnowledgeGraphEmbedding.evaluate()` は `self.dir_save` 配下の `test.txt` / `valid.txt` を探す仕様。

- before モデルは「モデルディレクトリに test/valid がある」前提で既存動作している
- after を `model_after_dir` に保存する場合、**`model_after_dir` に test/valid が無いと評価が失敗**し得る

対策（最小）:
- `updated_triples_dir/test.txt` と `updated_triples_dir/valid.txt` が存在する場合、
  それを `model_after_dir/test.txt` と `model_after_dir/valid.txt` にコピーする

※ より堅牢にするなら `IterationEvaluator` 側を拡張し、`kge.evaluate(test_triples=...)` を呼べるようにするが、
影響範囲が広いのでまずはコピー方式を採用する。

### 4.3 成果物
従来どおり `run_dir/retrain_eval/` へ出力する（互換維持）。

- `retrain_eval/updated_triples/`（既存）
- `retrain_eval/model_after/`（after 再学習結果を保存）
- `retrain_eval/evaluation/`（評価）
- `retrain_eval/summary.json`（拡張: after_mode, embedding_config, num_epochs, force_retrain 等を追記）

### 4.4 実装単位
- 変更対象: [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)
- 追加が望ましい小ユーティリティ（任意）
  - `simple_active_refine/io_utils.py` に `copy_if_exists(src, dst)` 相当を追加
  - もしくはスクリプト内で `shutil.copy2` を直に使う（最小変更優先ならこちら）

## 5. テスト方針（重要: 実学習を回す）

ユーザー要望により、テストでは「重くても」 **実際にKGEを学習/評価**して正しさを担保する。
このためモック/monkeypatch による `train_model()` の置き換えは行わない。

ただしテスト時間を抑えるため、データは最小のダミーKGを用い、epoch数は必ず 2 に固定する。

### 5.1 追加テスト（案）
- `tests/test_retrain_and_evaluate_after_arm_run_retrain_mode.py`

テスト内容（epoch=2で実行）:
1. tmp_path に `dataset_dir/` を作り、`train.txt`/`valid.txt`/`test.txt` に小さなトリプル集合（例: 5〜20本）を作成
2. `run_dir/iter_1/accepted_evidence_triples.tsv` を作成（train内に存在するトリプルでOK、または新規1本を混ぜる）
3. **beforeモデルを事前に学習**（epoch=2）し、`model_before_dir` に保存
  - `KnowledgeGraphEmbedding.train_model(dir_triples=dataset_dir, dir_save=model_before_dir, training_kwargs={... num_epochs:2 ...}, ...)`
  - embedding設定はテスト用に最小構成にする（`model=TransE`、embedding_dim小さめ等）
4. テスト対象のスクリプトを `--after_mode retrain --num_epochs 2` で実行
5. 検証:
  - `model_after_dir/trained_model.pkl` と `model_after_dir/training_triples/` が生成される
  - `model_after_dir/test.txt`（および `valid.txt` がある場合はそれも）が配置される
  - `run_dir/retrain_eval/summary.json` が生成され、`after_mode=retrain` と `num_epochs=2` が記録される
  - `run_dir/retrain_eval/evaluation/iteration_metrics.json` が生成される

## 6. Definition of Done (DoD)

- `--after_mode retrain` 実行で after の再学習が行われ、`model_after_dir/trained_model.pkl` と `training_triples/` が生成される
- `model_after_dir` に `test.txt`（あれば `valid.txt`）が配置され、評価が完走する
- `retrain_eval/summary.json` に after 学習設定が記録される
- テストで **epoch=2 の実学習** と評価が完走し、成果物が検証される

## 7. リスク・補足

- 追加evidenceが新規エンティティ/リレーションを含むと、after の entity/relation mapping が before と一致しない。
  - `score_triples()` が unknown を skip する挙動があるため、target triples が未知扱いになる可能性がある。
  - 実験データの前提（evidenceは既存KG由来）に依存するため、必要なら別途 mapping 互換性チェックを追加する。

## 8. 実装結果（2026-01-14 時点）

### 8.1 実装内容
- [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)
  - `--after_mode {load,retrain}`（default: retrain）を追加
  - retrain時は `updated_triples/` を入力に `KnowledgeGraphEmbedding.train_model(...)` を呼び、`model_after_dir/` に保存
  - 評価が動くように、`test.txt`/`valid.txt` を `model_after_dir/`（および必要なら `model_before_dir/`）へコピー
  - テストから呼べるよう `run(...)` 関数を追加（CLI main は run を呼ぶ）

注意:
- 本リポジトリの `KnowledgeGraphEmbedding.train_model()` は内部で `pykeen.pipeline.pipeline(**kwargs)` を呼ぶが、
  現環境の PyKEEN では `num_epochs` をトップレベル引数として受け取らないため、
  epochは `training_kwargs["num_epochs"]` で指定する。

### 8.2 テスト
- 追加: [tests/test_retrain_and_evaluate_after_arm_run_retrain_mode.py](../../tests/test_retrain_and_evaluate_after_arm_run_retrain_mode.py)
  - epoch=2 で before を事前学習 → スクリプトで after を updated_triples から再学習 → 評価成果物生成まで検証

### 8.3 pytest成果物の保持とログ表示
- [pytest.ini](../../pytest.ini) を更新
  - `--basetemp=tmp/pytest-of-root` により、tmp_path は `/app/tmp/pytest-of-root/` 配下へ作成
  - `tmp_path_retention_policy = all` により、テスト終了後も成果物を保持
  - `log_cli = true` / `log_cli_level = INFO` により、コンソールにログを表示

例（直近テスト実行時）:
- afterモデル: `/app/tmp/pytest-of-root/test_retrain_and_evaluate_afte0/run/retrain_eval/model_after/trained_model.pkl`
