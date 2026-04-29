# RULE-20260117-INITIAL_RULE_POOL-001: build_initial_rule_pool.py 実装仕様（初期ルールプール生成）

作成日: 2026-01-17
最終更新日: 2026-01-17

---

## 0. 目的とスコープ

`build_initial_rule_pool.py` は、**学習済みKGEモデル**と **ターゲット関係**を入力として、AMIE+由来のHornルール集合から「初期ルールプール」を構築し、後続（arm生成・arm-run）のために保存する。

本仕様は、現行実装（2026-01-17時点）の `build_initial_rule_pool.py` が提供する **関数API**と **CLI**、入出力ファイル、例外/終了条件を再実装可能な粒度で定義する。

非スコープ:
- AMIE+の内部アルゴリズム詳細
- ルール品質指標の意味の解説（support/head_coverage/pca_conf 等）

---

## 1. 提供API

### 1.1 関数: `build_initial_rule_pool(...)`

シグネチャ（概念）:

- 入力:
  - `model_dir: str`
  - `target_relation: str`
  - `output_dir: str`
  - `n_rules: int`
  - `sort_by: str`
  - `mode: str`（`"entire"` / `"high-score"`）
  - `min_head_coverage: float`
  - `min_pca_conf: float`
  - `lower_percentile: float`（high-score時のみ）
  - `k_neighbor: int`（high-score時のみ）
- 出力（戻り値）:
  - `(csv_path: str, pkl_path: str, summary_path: str)`

役割:
- `KnowledgeGraphEmbedding(model_dir=...)` をロードし、AMIE+ルール抽出を実行
- `BaseRuleGenerator.create_initial_rule_pool_from_amie(...)` により、抽出ルールから上位 `n_rules` を選別してルールプール化
- ルールプールを `output_dir` へ `csv/pkl/txt` として保存

---

## 2. 入力仕様

### 2.1 `model_dir`（必須）

`simple_active_refine.embedding.KnowledgeGraphEmbedding` がロード可能な「学習済みKGEディレクトリ」。

最低限の前提（KGEラッパの期待）:
- `trained_model.pkl`
- `training_triples/`

※ このスクリプト自身は厳密検証を行わず、ロードに失敗した場合は例外で落ちる。

### 2.2 `target_relation`（必須）

AMIE+抽出で「head relation」として扱う対象関係（例: `/people/person/nationality`）。

### 2.3 `mode` と抽出戦略

- `mode="entire"`:
  - `extract_rules_from_entire_graph(...)` を使用
  - KG全体（埋め込みに紐づく triples）からルール抽出
- `mode="high-score"`:
  - `extract_rules_from_high_score_triples(...)` を使用
  - `lower_percentile` 以上の高スコアトリプル起点で k-hop 囲い込みサブグラフからルール抽出

### 2.4 抽出ルール品質の下限

両モード共通で次をAMIE+抽出のフィルタとして渡す。
- `min_head_coverage`
- `min_pca_conf`

---

## 3. 出力仕様（ファイル/ディレクトリ）

`output_dir` 配下に次を生成する。

- `initial_rule_pool.csv`
  - `AmieRules.to_csv()` の出力
  - AMIE+メトリクス付きルール一覧
- `initial_rule_pool.pkl`
  - `AmieRules.to_pickle()` の出力（pickle）
  - 後段（arm生成・arm-run）が直接ロードする主成果物
- `initial_rule_pool.txt`
  - `_summarize_rule_pool(...)` による人間可読サマリ
- `amie_tmp/`
  - AMIE+実行の作業ディレクトリ

---

## 4. 処理フロー（アルゴリズム）

1. `output_dir` を作成（`mkdir(parents=True, exist_ok=True)`）
2. `KnowledgeGraphEmbedding(model_dir=model_dir)` をロード
3. `output_dir/amie_tmp` を作成
4. `mode` に応じて AMIE+ ルール抽出
   - entire: `extract_rules_from_entire_graph(...)`
   - high-score: `extract_rules_from_high_score_triples(...)`
5. `amie_rules.rules` が空なら `RuntimeError` で停止
6. `BaseRuleGenerator().create_initial_rule_pool_from_amie(amie_rules, n_rules, sort_by)` を実行
7. `csv/pkl/txt` を保存し、パスを返す

---

## 5. 失敗時の挙動

- AMIE+抽出の結果が0件の場合: `RuntimeError("AMIE+ did not return any rules...")`
- `KnowledgeGraphEmbedding` のロード失敗: 例外伝播（呼び出し元へ）
- I/O（書き込み失敗など）: 例外伝播

---

## 6. CLI仕様

コマンド:
- `python build_initial_rule_pool.py ...`

必須:
- `--model-dir`
- `--target-relation`

主なオプション:
- `--output-dir`（デフォルト `./tmp/initial_rule_pool`）
- `--n-rules`（デフォルト 20）
- `--sort-by`（デフォルト `support`）
- `--mode {entire,high-score}`（デフォルト `entire`）
- `--min-head-coverage`（デフォルト 0.01）
- `--min-pca-conf`（デフォルト 0.01）
- `--lower-percentile`（デフォルト 90.0, high-scoreのみ）
- `--k-neighbor`（デフォルト 1, high-scoreのみ）

---

## 7. 実装参照（ソースコード）

- 実装: [build_initial_rule_pool.py](../../build_initial_rule_pool.py)
- 依存: [simple_active_refine/rule_extractor.py](../../simple_active_refine/rule_extractor.py)
- 依存: [simple_active_refine/rule_generator.py](../../simple_active_refine/rule_generator.py)
- 依存: [simple_active_refine/embedding.py](../../simple_active_refine/embedding.py)

---

## 8. 参考文献（docs）

- 統合ランナー仕様（この出力が入力になる）: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- v3パイプライン概要（抽出・評価の位置づけ）: [docs/rules/RULE-20260111-PIPELINE_OVERVIEW-001.md](RULE-20260111-PIPELINE_OVERVIEW-001.md)
- arm/バンディット設計（前提）: [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
