````markdown
# REC-20260111-ARM_REFINE-005: Arm数削減のためのRule Pool事前フィルタ（実装計画）

作成日: 2026-01-11
更新日: 2026-01-11
参照:
- [docs/records/REC-20260111-ARM_REFINE-004.md](REC-20260111-ARM_REFINE-004.md)
- [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- 実装対象: [build_initial_arms.py](../../build_initial_arms.py), [simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py), [simple_active_refine/amie.py](../../simple_active_refine/amie.py)

---

## 0. 背景 / 課題

20iterの検証（REC-004）では、
- 20iter × 2arm/iter = 40回の選択がすべてユニーク arm（再選択0）

となり、「armごとのreward推移（時系列）」を観測できない。

この挙動の主要因は、初期armプールが大きく、selector（UCB）が探索フェーズから抜けにくいこと。
特に、現状の初期arm生成は
- singleton arm = ルール数
- pair arm = 上位 `k_pairs`（共起Jaccard）

なので、**ルール数がそのままarm数を支配する**。

よって、arm数削減の第一手は「arm生成前に rule_pool を絞り込む」こと。

---

## 1. 目的

- 初期armプールのサイズを下げ、同一armの再選択を起こしやすくする
- 20iter程度でも armごとのreward推移（収束/劣化/安定）を観測可能にする
- 絞り込み条件を再現可能（CLI引数/設定ファイルで固定）にする

---

## 2. 現状（仕様整理）

### 2.1 初期arm生成の現状

- 実装: [simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py)
- singleton: rule_poolの全ルールから生成（絞り込み無し）
- pair: supported-target集合のJaccard（`cooc`）を計算し、上位 `k_pairs` のみ採用

### 2.2 ルールの品質指標とフィルタ

- `AmieRule`の主な指標フィールド（[simple_active_refine/amie.py](../../simple_active_refine/amie.py)）:
  - `support`, `std_conf`, `pca_conf`, `head_coverage`, `body_size`, `pca_body_size`
- 既存の絞り込み機構:
  - `AmieRules.filter(min_pca_conf, min_head_coverage, sort_by, top_k, **llm_kwargs)`
  - `AmieRules.exclude_relations_by_pattern(exclude_patterns)`
- 既存config: [config_rule_filter.json](../../config_rule_filter.json)

---

## 3. 実装方針

### 3.1 基本方針（B案）

1) `build_initial_arms.py` の入力 `rule_pool` を、arm生成前に prefilter する
2) prefilter後のルールだけで singleton/pair arm を生成する

期待される効果:
- singleton armが直接減る（= 最も効く）
- pair候補も減り、探索の分散が減って再選択が起きやすくなる

### 3.2 prefilter のデフォルト（推奨）

- まずは **LLMなし**（再現性・コスト観点）
- `sort_by` は `pca_conf` または `head_coverage` を優先
- `top_k` を 50 / 100 / 200 あたりで振って観測

LLMを使う場合は二段構え:
- 数値指標で軽く落とす（min_*）→残りを `sort_by=llm` で top_k

---

## 4. 実装タスク（コード変更）

### 4.1 `build_initial_arms.py` の拡張

- 追加CLI（案）
  - `--rule-top-k`（int, default: None）
  - `--rule-sort-by`（str, default: pca_conf など。`llm` も許可）
  - `--min-pca-conf` / `--min-head-coverage`（float, default: None）
  - `--exclude-relation-pattern`（複数指定可）
  - `--rule-filter-config`（json, 任意。`config_rule_filter.json` 互換）

- 処理順（案）
  1) `AmieRules.from_pickle(rule_pool_path)`
  2) exclude（指定時）
  3) `rule_pool.filter(...)` で prefilter
  4) `build_initial_arms(rule_pool.rules, ...)`

- ログ/出力（必須）
  - `rules_before`, `rules_after`, `singleton_count`, `pair_count` をINFOで出す
  - `initial_arms.txt` にも「ルールフィルタ条件」をヘッダとして出す（再現性）

### 4.2 テスト追加

- `tests/test_build_initial_arms_rule_prefilter.py`（新規）
  - ダミー `AmieRules`（手書き）を作り、top_k/閾値で絞れること
  - 絞り込み後の `initial_arms.pkl` の singleton数が想定通り減ること
  - exclude pattern が head/body の relation に効くこと

---

## 5. 検証計画

### 5.1 prefilterの妥当性（静的確認）

- `build_initial_arms.py` 実行後に以下を比較
  - `len(rule_pool)`（before/after）
  - `len(singleton_arms)`
  - `len(pair_arms)`

### 5.2 再選択（動的確認）

- 絞り込んだ `initial_arms.pkl` を使って、`run_arm_refinement.py` を 20iter 実行
- 観測指標
  - 選択armのユニーク数 / 再選択回数
  - 同一armの reward 時系列（複数回観測できること）
  - evidence追加数の推移

---

## 6. リスク / 注意

- `rule_keys` が `str(AmieRule)` 依存のため、将来的に表示形式が変わると互換が揺らぐ（別途ルールID導入が望ましい）。
- prefilterを強くし過ぎると、armが少なすぎて探索が偏り、局所解に陥る可能性がある。

````
