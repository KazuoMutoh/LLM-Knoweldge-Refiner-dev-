# REC-20260114-ARM_PIPELINE-001: 初期ルールプール→初期arm→反復精錬→（再学習/評価）を連続実行する統合ランナー 実装計画

作成日: 2026-01-14  
最終更新日: 2026-01-14

更新日: 2026-01-14
- `retrain_and_evaluate_after_arm_run.py` が after 再学習（`--after_mode retrain`）に対応したため、統合ランナー計画の「再学習ギャップ」記述と引数設計を更新。

更新日: 2026-01-14
- 実装完了: `run_full_arm_pipeline.py` を追加し、最小ユニットテストを追加（`tests/test_run_full_arm_pipeline.py`）。
- `run_arm_refinement.py` に import-safe な `run(...)` 関数を追加（CLI `main()` は `run(...)` を呼ぶ）。

## 0. 目的

下記の既存CLIスクリプトを、同一の run ディレクトリ配下で **連続実行**できるようにする。

- [build_initial_rule_pool.py](../../build_initial_rule_pool.py)
- [build_initial_arms.py](../../build_initial_arms.py)
- [run_arm_refinement.py](../../run_arm_refinement.py)
- [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)

狙い:
- 「実験（1回の試行）」を 1 コマンドで起動できるようにし、再現性と運用性を上げる
- 途中失敗時に、作業者が手でパスをつなぎ替えずに再実行/再開できるようにする

## 1. 前提（現状のI/F整理）

### 1.1 build_initial_rule_pool.py
入力:
- `--model-dir`（学習済みKGEディレクトリ。`trained_model.pkl` を含む）
- `--target-relation`（例: `/people/person/nationality`）
- `--output-dir`（出力先）
- 任意の抽出/フィルタ設定（`--mode`, `--min-head-coverage`, `--min-pca-conf`, `--lower-percentile`, `--k-neighbor` 等）

出力（output-dir 配下）:
- `initial_rule_pool.csv`
- `initial_rule_pool.pkl`
- `initial_rule_pool.txt`

### 1.2 build_initial_arms.py
入力:
- `--rule-pool`（`initial_rule_pool.pkl`）
- `--target-triples`（`target_triples.txt`）
- 候補集合（どちらか必須）
  - `--candidate-triples`（例: `train_removed.txt`）
  - `--dir-triples`（`train.txt` と任意で `train_removed.txt` を読む。`--include-train-removed`）
- arm生成設定（`--k-pairs`, `--max-witness-per-head` 等）
- ルール事前フィルタ（任意）
  - `--rule-top-k`, `--rule-sort-by`, `--min-pca-conf`, `--min-head-coverage`, `--exclude-relation-pattern`, `--rule-filter-config`

出力（output-dir 配下）:
- `initial_arms.json`
- `initial_arms.pkl`
- `initial_arms.txt`

### 1.3 run_arm_refinement.py
入力:
- `--base_output_path`（反復出力の基底。`iter_k/` が作られる）
- `--initial_arms`（`initial_arms.json` または `initial_arms.pkl`）
- `--rule_pool_pkl`（`initial_rule_pool.pkl`）
- `--dir_triples`（`train.txt` を含むディレクトリ）
- `--target_triples`（`target_triples.txt`）
- `--candidate_triples`（例: `train_removed.txt`）
- 反復/選択/評価の設定（`--n_iter`, `--k_sel`, `--n_targets_per_arm`, `--selector_strategy` 等）

出力:
- `base_output_path/iter_*/...`（各 iter の `accepted_evidence_triples.tsv` 等）

### 1.4 retrain_and_evaluate_after_arm_run.py
重要: 現状の実装は、`--after_mode` により **「評価のみ（load）」と「after再学習+評価（retrain）」の両方**をサポートする。

- `--after_mode load`: 学習済みモデル before/after を読み込んで評価する
- `--after_mode retrain`: before は学習済みモデルを読み込み、after は `updated_triples/`（train ∪ accepted evidence）で再学習して評価する

追記（2026-01-14）:
- 統合ランナーから import で安全に呼べる `run(...)` 関数が追加されている

入力:
- `--run_dir`（arm-run 出力ディレクトリ。`iter_*/accepted_evidence_triples.tsv` を集約）
- `--dataset_dir`（元 dataset: `train/valid/test`）
- `--target_triples`
- `--after_mode {load,retrain}`（after の扱い。`load`=学習済みモデル読込、`retrain`=updated_triples で再学習。デフォルトは `retrain`）
- 任意: `--model_before_dir`, `--model_after_dir`（省略時は `<run_dir>/retrain_eval/model_before|after`）
- 任意（`--after_mode retrain` のとき使用）: `--embedding_config`（例: `./config_embeddings.json`）
- 任意（`--after_mode retrain` のとき使用）: `--num_epochs`（再学習のエポック数）
- 任意（`--after_mode retrain` のとき使用）: `--force_retrain`（既存の after モデルがあっても再学習する）
- 任意: `--exclude_predicate`（追加evidenceから除外するpredicate）

出力（run_dir/retrain_eval 配下）:
- `updated_triples/`（train=original ∪ evidence）
- `evaluation/`（Hits@k/MRR + target score）
- `summary.json`
- `model_before/`（評価に用いる before モデル。指定がなければ `<run_dir>/retrain_eval/model_before`）
- `model_after/`（評価に用いる after モデル。`--after_mode retrain` の場合はここに再学習結果を保存し、`trained_model.pkl` を生成）

## 2. 実装方針（統合ランナー）

### 2.1 新規スクリプトの追加
リポジトリ直下に、統合ランナーを 1 つ追加する。

候補ファイル名:
- `run_full_arm_pipeline.py`

責務:
- 4本のスクリプトを「決まった順序で」起動し、出力パスを次工程へ受け渡す
- 実験の run directory を 1 箇所に集約し、成果物の配置を安定化する

実装方式（優先順位）:
1. **importして関数呼び出し**（同一プロセスで `build_initial_rule_pool.build_initial_rule_pool(...)` 等を呼ぶ）
   - メリット: 引数の受け渡しが安全、ログ/例外の扱いが統一しやすい
2. 必要に応じて **subprocess**（CLIとして起動）
   - メリット: CLIと同じ挙動で実行できる
   - 注意: シェルエスケープ/環境差、ログ収集が手間

本件は既存スクリプトが関数を公開しているため、(1) を第一候補とする。
- `build_initial_rule_pool.py`: `build_initial_rule_pool(...)`
- `build_initial_arms.py`: `build_initial_arm_pool(...)`
- `run_arm_refinement.py`: `main()`（関数化されているが、内部で argparse を読む。import実行時は注意）
- `retrain_and_evaluate_after_arm_run.py`: `main()`（同上）

※ `run_arm_refinement.py` / `retrain_and_evaluate_after_arm_run.py` は import 呼び出しだけだと argparse が邪魔になるため、
「コア処理関数（引数を受け取る）」を追加して main はそれを呼ぶ形に寄せるのが安全。

追記（2026-01-14）:
- `retrain_and_evaluate_after_arm_run.py` は `run(...)` 関数が追加され、統合ランナーから import で安全に呼べる前提が整った。

### 2.2 run ディレクトリ構造（成果物の置き場）
統合ランナーの `--run_dir` を 1 つ受け、配下に工程別ディレクトリを固定する。

例:
- `<run_dir>/00_rule_pool/`
- `<run_dir>/01_arms/`
- `<run_dir>/02_arm_run/`（ここが `run_arm_refinement.py --base_output_path` になる）
- `<run_dir>/03_retrain_eval/`（既存スクリプトは `<run_dir>/retrain_eval` を使うので、整合のためどちらかに寄せる）

方針:
- 既存スクリプトの出力仕様を壊さないことを優先
- `retrain_and_evaluate_after_arm_run.py` のデフォルトが `<run_dir>/retrain_eval` のため、
  統合ランナー側の `--run_dir` は **arm-runの base 出力ディレクトリ**（`02_arm_run`）にするか、
  もしくは評価スクリプト側の出力先を指定可能にして `03_retrain_eval` に出す

最小変更案:
- 統合ランナーの run_dir を「実験ルート」とし、
  - arm-run は `<run_dir>/arm_run/` を `--base_output_path` にする
  - 評価は `--run_dir <run_dir>/arm_run/` を渡す（既存の仕様に合う）

### 2.3 連続実行の再開（idempotency）
統合ランナーは「途中まで生成済みならスキップできる」ようにする。

- Step 1（rule pool）: `<run_dir>/rule_pool/initial_rule_pool.pkl` があれば再生成しない
- Step 2（arms）: `<run_dir>/arms/initial_arms.json` があれば再生成しない
- Step 3（arm-run）: `<run_dir>/arm_run/iter_1/` 等があれば警告し、`--force` 指定時のみ上書き
- Step 4（eval）: `<run_dir>/arm_run/retrain_eval/summary.json` があればスキップ

追加オプション案:
- `--start-from {rule_pool,arms,arm_run,eval}`
- `--stop-after {rule_pool,arms,arm_run,eval}`
- `--force`（既存成果物を削除/上書きして実行）

※ 仕様が増えすぎないよう、まずは `--force` と「成果物存在チェックによるスキップ」だけで開始してよい。

## 3. 統合ランナー CLI 設計（案）

### 3.1 必須引数
- `--run_dir`: 実験ルートディレクトリ（工程成果物の集約先）
- `--model_dir`: build_initial_rule_pool 用
- `--target_relation`: build_initial_rule_pool 用
- `--dataset_dir`: 元データセット（`train/valid/test`）
- `--target_triples`: 対象トリプル（arm生成と評価で利用）
- `--candidate_triples`: 候補トリプル（arm生成・arm-runで利用）

### 3.2 任意引数（そのまま各工程に渡す）
- rule pool 抽出/フィルタ系（`--mode`, `--min_head_coverage`, ...）
- arm生成系（`--k_pairs`, `--max_witness_per_head`, `--rule_filter_config` 等）
- arm-run 系（`--n_iter`, `--k_sel`, `--selector_strategy`, ...）
- eval/再学習系（`--after_mode load|retrain`, `--embedding_config`, `--num_epochs`, `--force_retrain`, `--model_before_dir`, `--model_after_dir`, `--exclude_predicate`）

実装メモ:
- 「統合ランナーの引数」と「各工程の引数」が二重に増えるのを避けるため、
  まずは統合ランナーで **必要最小の引数のみ**を受け、
  詳細は json config を渡す方式も検討する（ただし本件はまず最小実装が良い）。

## 4. 再学習（retrain）の扱い：統合ランナーでの実行パス

現状の [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py) は、評価のみ（load）だけでなく
**afterの再学習（retrain）** に対応している。

統合ランナーは、次の2パスをサポートする。

- パスA（評価のみ）: `--after_mode load`
  - before/after ともに学習済みモデルを読み込んで評価する
  - 事前に `--model_before_dir` と `--model_after_dir` を用意する運用

- パスB（after再学習 + 評価）: `--after_mode retrain`（推奨デフォルト）
  - before は学習済みモデルを読み込む
  - after は `updated_triples/`（train ∪ accepted evidence）で再学習して `model_after_dir` に保存し、そのモデルで評価する
  - 学習設定は `--embedding_config`（例: `./config_embeddings.json`）と `--num_epochs` で指定する

備考:
- 反復中に再学習しない（arm-run完了後に一度だけ学習）ため、combo-banditの設計原則とは矛盾しない。

## 5. 実装ステップ（DoD付き）

### Step 1: 統合ランナー骨格
- `run_full_arm_pipeline.py` を追加
- `--run_dir` を受け、工程別の出力ディレクトリを作成

DoD:
- `--help` が表示できる
- `--run_dir` 配下に `rule_pool/arms/arm_run/` が作られる

### Step 2: Step1（rule pool）を接続
- `build_initial_rule_pool.build_initial_rule_pool()` を呼び、`rule_pool/initial_rule_pool.pkl` を得る

DoD:
- 既存成果物がある場合はスキップできる

### Step 3: Step2（arms）を接続
- `build_initial_arms.build_initial_arm_pool()` を呼び、`arms/initial_arms.json` を得る

DoD:
- 既存成果物がある場合はスキップできる

### Step 4: Step3（arm refinement）を接続
- `run_arm_refinement.py` を import 実行できる形へ寄せる（argparse依存を分離）
  - 例: `run_arm_refinement.run(...)` のような関数を新設（CLI `main()` は `run(...)` を呼ぶ）

DoD:
- 統合ランナーから arm-run が起動でき、`arm_run/iter_*/accepted_evidence_triples.tsv` が出る

### Step 5: Step4（eval）を接続
- `retrain_and_evaluate_after_arm_run.py` の `run(...)` を統合ランナーから呼び出す
- `--after_mode` に応じて「評価のみ」または「after再学習+評価」を切り替える

DoD:
- `arm_run/retrain_eval/summary.json` が生成される
- `--after_mode retrain` の場合は `arm_run/retrain_eval/model_after/trained_model.pkl` が生成される
- `--after_mode load` の場合は学習済モデル不足で warning + exit できる

### Step 6: 最小のユニットテスト
- 統合ランナー自体は外部依存が大きいので、
  - 「パス解決と存在チェックロジック」
  - 「工程間の成果物パスの受け渡し」
 だけをダミーでテストする（I/Oは tmp を使用）

DoD:
- `pytest` で統合ランナーの小テストが通る

## 6. 期待される利用例（例示）

- 実験ルート: `experiments/20260114/my_run_001/`
- 入力:
  - `--model_dir models/.../`
  - `--dataset_dir experiments/test_data_for_nationality_v3/`
  - `--candidate_triples experiments/test_data_for_nationality_v3/train_removed.txt`
  - `--target_triples experiments/test_data_for_nationality_v3/target_triples.txt`

※ 正確なコマンド例は、統合ランナーを実装後に README へ追記する。
