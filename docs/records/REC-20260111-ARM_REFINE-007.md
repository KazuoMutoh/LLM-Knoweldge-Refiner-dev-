# REC-20260111-ARM_REFINE-007: arm-run追加evidence反映後のKGE評価（学習済モデル使用） 実装計画

作成日: 2026-01-11  
最終更新日: 2026-01-12

更新日: 2026-01-12
- 方針変更: 本フローでは **KGEの再学習は行わず**、学習済モデル（before/after）を読み込んで評価する。
- 学習済モデルが指定ディレクトリに存在しない場合は **警告を出して処理を停止**する。

## 背景と目的
v3 arm refinement の出力（各iterationで受理された `accepted_evidence_triples.tsv`）を元データセット（train/valid/test）へ反映し、
updated dataset を構築した上で、**学習済KGE（before/after）を用いて精度再評価（Hits@k/MRR）** と **target triple score** を比較するための実装計画。

既存の学習・評価ロジックを最大限再利用し、最小の追加実装で end-to-end を実現する。

参照:
- [REC-20260111-ARM_REFINE-006.md](REC-20260111-ARM_REFINE-006.md): arm=10/iter=100 実験と追加evidenceの集計
- `KnowledgeGraphEmbedding`（`simple_active_refine/embedding.py`）
- `IterationEvaluator`（`simple_active_refine/evaluation.py`）

## 方針（重要）
- **「arm-runでKGに足したものはそのまま」方針**を採用する。
  - `accepted_evidence_triples.tsv` に含まれるトリプル（predicateが `/people/person/nationality` を含む可能性がある）も、
    更新trainへ **含める** のをデフォルトとする。
- ただし、比較実験・漏洩懸念の切り分けのために、任意で除外できるオプションは用意する（デフォルトOFF）。

## 成果物（作るもの）
### 1) 新規CLI（最小）
- 新規スクリプト: `retrain_and_evaluate_after_arm_run.py`（リポジトリ直下）
- 役割: 
  1. arm-run出力から追加evidenceを集約
  2. updated dataset（train/valid/test）を作成
  3. before/after の **学習済KGEモデル** を読み込み
  4. before/after の評価（Hits@k/MRR + target score）を実行
  5. 結果を run_dir 配下に保存（学習済モデルが無い場合は警告して停止）

### 2) 追加のユーティリティ（必要最小限）
- 追加evidence集約・dataset更新の関数を `simple_active_refine/io_utils.py` もしくは新規 `simple_active_refine/dataset_update.py` に配置
  - 既存に相当機能がある場合はそれを優先して再利用

### 3) テスト
- `tests/test_dataset_update_after_arm_run.py`
  - 小さいダミー `run_dir/iter_k/accepted_evidence_triples.tsv` から `updated_triples/train.txt` が正しく構築されること
  - `--exclude-predicate` が効くこと（デフォルトは除外しない）

## 入出力仕様
### 入力
- `--run_dir`: arm-run出力ディレクトリ
  - 例: `experiments/20260111/v3_arm_run_nationality_v3_100iter_arm10_nt5`
  - 各 `iter_k/accepted_evidence_triples.tsv` を参照
- `--dataset_dir`: 元データセットディレクトリ
  - `train.txt/valid.txt/test.txt` を含む
  - 例: `experiments/test_data_for_nationality_v3`
- `--target_triples`: target triple list（score比較）
  - 例: `experiments/test_data_for_nationality_v3/target_triples.txt`
- `--model_before_dir`: 学習済モデル（before）のディレクトリ
  - デフォルト: `run_dir/retrain_eval/model_before`
- `--model_after_dir`: 学習済モデル（after）のディレクトリ
  - デフォルト: `run_dir/retrain_eval/model_after`

### 出力（run_dir配下に保存）
- `retrain_eval/`
  - `updated_triples/`
    - `train.txt`（更新後）
    - `valid.txt` / `test.txt`（原則コピー）
    - `added_evidence.tsv`（集約した追加分、重複除去後）
  - `evaluation/`
    - `iteration_metrics.json`
    - `iteration_evaluation.md`

## 実装ステップ
### Step 1: 追加evidence集約
- `run_dir/iter_*/accepted_evidence_triples.tsv` を全て読み込む
- `[(s,p,o)]` の集合として重複除去
- `added_evidence.tsv` として保存

注意:
- `arm_history.json` は累積形式のため、ここでは参照しない（追加evidenceの単一ソースを `accepted_evidence_triples.tsv` に固定）。

### Step 2: updated dataset作成
- `train_updated = train_original ∪ added_evidence`
- `valid.txt` / `test.txt` はそのままコピー
- 保存先: `run_dir/retrain_eval/updated_triples/`

オプション:
- `--exclude-predicate`（複数指定可）
  - 指定されたpredicateを持つ追加evidenceを除外してtrainを作る
  - デフォルト: 指定なし（=除外しない）

### Step 3: 学習済KGEの読み込み（before/after）
- `KnowledgeGraphEmbedding(model_dir)` を使用
  - before: `model_before_dir`
  - after: `model_after_dir`
- 学習済モデルが存在しない場合は警告して停止する。

### Step 4: 評価（before/after）
- `IterationEvaluator.evaluate_iteration()` を流用して、iteration=1相当の before/after 比較として扱う
  - `kge_before = KnowledgeGraphEmbedding(model_before_dir)`
  - `kge_after  = KnowledgeGraphEmbedding(model_after_dir)`
  - `target_triples` を読み込んで score比較
  - `n_triples_added = len(added_evidence_unique)`
- `evaluation/` に JSON + Markdown を保存

## 受け入れ基準（Definition of Done）
- CLI 1コマンドで end-to-end（集約→更新train→学習済モデル読み込み→評価）が完走する
- `retrain_eval/` 配下に updated_triples / evaluation が出力される（モデルは入力として参照）
- テストが追加され、少なくとも dataset更新ロジックが検証される

## リスクと補足
- `/people/person/nationality` を追加evidenceとしてtrainに含めると、
  評価対象の分割（valid/test）との関係によっては“簡単化”が起きる可能性がある。
  本RECの方針では含めるが、将来的な比較のため `--exclude-predicate` を残す。
