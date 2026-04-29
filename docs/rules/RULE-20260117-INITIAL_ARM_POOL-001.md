# RULE-20260117-INITIAL_ARM_POOL-001: build_initial_arms.py 実装仕様（初期armプール生成）

作成日: 2026-01-17
最終更新日: 2026-01-24
更新日: 2026-01-24（pair-arm の support/Jaccard 計算母集団を candidate/train で切替可能にし、既定値は candidate を維持）

---

## 0. 目的とスコープ

`build_initial_arms.py` は、初期ルールプール（AMIE+由来のルール集合）と、ターゲット/候補トリプルを入力として、
**arm（ルールの組）**の初期プールを構築し保存する。

本仕様は、現行実装（2026-01-17時点）の `build_initial_arms.py` が提供する **関数API**と **CLI**、入出力ファイル、フィルタ設定、例外/終了条件を再実装可能な粒度で定義する。

---

## 1. 提供API

### 1.1 関数: `build_initial_arm_pool(...)`

戻り値:
- `(json_path: str, pkl_path: str, summary_path: str)`

主責務:
- `rule_pool_path` からルールをロード（pickle）
- （任意）ルール事前フィルタでルール数を削減（singleton arm肥大化対策）
- `target_triples` と `candidate_triples` をロード
- `simple_active_refine.arm_builder.build_initial_arms(...)` により arm を生成
- `initial_arms.json/.pkl/.txt` を保存

---

## 2. 入力仕様

### 2.1 ルールプール: `rule_pool_path`（必須）

- `build_initial_rule_pool.py` が生成する `initial_rule_pool.pkl`
- `AmieRules.from_pickle(rule_pool_path)` でロードされる

### 2.2 ターゲットトリプル: `target_triples_path`（必須）

- TSV 3列: `head<TAB>relation<TAB>tail`
- 不正行（3列でない）はスキップ

### 2.3 候補トリプル（次のいずれか必須）

1) `candidate_triples_path` を直接渡す
- TSV 3列

2) `dir_triples` を渡す（`candidate_triples_path=None` の場合）
- `dir_triples/train.txt` を必須
- `include_train_removed=True` の場合、`dir_triples/train_removed.txt` が存在すれば連結して候補集合に含める

注意（統合ランナーとの関係）:
- 統合ランナーは `candidate_triples_path`（典型: train_removed.txt）を渡す。
  - `dir_triples` には `dataset_dir` を渡す（pair-arm の support source を train に切り替える場合に train.txt を参照するため）。
  - 仕様: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)

### 2.4 pair-arm 用 support triples（任意）

pair-arm（ルールペア）選別では、ターゲット支持集合の共起（Jaccard）を計算するために
「rule body を満たすための探索母集団（support triples）」を用いる。

実装は `pair_support_source` により次を切替可能:
- `candidate`（既定）: `candidate_triples` を母集団にする（後方互換。典型: train_removed.txt）
- `train`: `train.txt` のみを母集団にする（候補未知という問題設定に合わせたい場合）

`pair_support_source=train` の場合、母集団は次のいずれかで与える:
- `--pair-support-triples`（明示ファイル）
- `--dir-triples` の `train.txt`（暗黙）

---

## 3. ルール事前フィルタ（任意）

目的:
- singleton arm（ルール1本のarm）が多すぎると探索が破綻しやすいため、ルール集合を先に縮小する。

適用順序（実装準拠）:
1. `rule_filter_config_path` が指定されていればJSONを読み、以下を上書きしうる
   - `top_k`（→ `rule_top_k`）
   - `use_llm`（trueなら `rule_sort_by="llm"` に変更）
   - `min_head_coverage`, `min_pca_conf`
2. `exclude_relation_patterns` が指定されていれば `AmieRules.exclude_relations_by_pattern(...)` を適用
3. `rule_top_k` の指定があれば `AmieRules.filter(..., sort_by=rule_sort_by, top_k=rule_top_k)` を適用
   - `rule_top_k` が無くても `min_*` が指定されていれば top_k=len(rules) で filter を適用

備考:
- `rule_sort_by` は `support|std_conf|pca_conf|head_coverage|body_size|pca_body_size|llm` を想定

---

## 4. arm生成アルゴリズム（委譲先の契約）

arm生成自体は `simple_active_refine.arm_builder.build_initial_arms(...)` に委譲される。

本スクリプトが渡す内容:
- ルール列: `rule_pool.rules`
- ターゲットトリプル列
- 候補トリプル列
- `ArmBuilderConfig(k_pairs, max_witness_per_head, pair_support_source)`
- （`pair_support_source=train` の場合）`pair_support_triples`（train.txt 由来、または明示ファイル）

（実装上の期待）:
- singleton arm を全ルールについて作成
- さらに、ターゲット支持集合の共起（Jaccard等）に基づき上位 `k_pairs` のペアarmを作成

---

## 5. 出力仕様

`output_dir` 配下に次を保存:

- `initial_arms.json`
  - list[object] 形式
  - 各要素: `{ "arm_type": str, "rule_keys": list[str], "metadata": object }`
- `initial_arms.pkl`
  - `Arm` オブジェクトのlistをpickle保存
- `initial_arms.txt`
  - `_summarize(...)` による簡易サマリ（1arm/1行）

---

## 6. 失敗時の挙動

- `candidate_triples_path` 未指定かつ `dir_triples` 未指定: `ValueError`
- `dir_triples/train.txt` が無い: `FileNotFoundError`
- `rule_filter_config_path` が dict 以外: `ValueError`
- その他（ファイル読み書き）: 例外伝播

---

## 7. CLI仕様

コマンド:
- `python build_initial_arms.py ...`

必須:
- `--rule-pool`
- `--target-triples`

候補集合:
- `--candidate-triples` または `--dir-triples` のどちらか
- `--include-train-removed` は `--dir-triples` 利用時のみ意味を持つ

主なオプション:
- `--output-dir`（デフォルト `./tmp/initial_arms`）
- `--k-pairs`（デフォルト 20）
- `--max-witness-per-head`（デフォルト None）

pair-arm support source:
- `--pair-support-source {candidate,train}`（デフォルト `candidate`）
- `--pair-support-triples`（`pair-support-source=train` のときのみ使用。省略時は `<dir_triples>/train.txt`）

ルール事前フィルタ:
- `--rule-top-k`
- `--rule-sort-by`（デフォルト `pca_conf`）
- `--min-pca-conf`, `--min-head-coverage`
- `--exclude-relation-pattern`（repeatable）
- `--rule-filter-config`（JSON。`config_rule_filter.json`互換）

---

## 8. 実装参照（ソースコード）

- 実装: [build_initial_arms.py](../../build_initial_arms.py)
- 依存: [simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py)
- 依存: [simple_active_refine/amie.py](../../simple_active_refine/amie.py)

---

## 9. 参考文献（docs）

- 統合ランナー仕様（この出力が入力になる）: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- arm設計（witness/衝突、出力規約）: [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- 統合ランナー設計記録（ルール事前フィルタの位置づけ）: [docs/records/REC-20260114-ARM_PIPELINE-001.md](../records/REC-20260114-ARM_PIPELINE-001.md)
