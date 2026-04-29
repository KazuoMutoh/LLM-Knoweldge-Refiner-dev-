# REC-20260111-ARM_REFINE-002: 反復精錬（armベース）実装計画 v1（実行コード実装）

作成日: 2026-01-11
更新日: 2026-01-11
参照:
- [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- [docs/records/REC-20260111-ARM-REFINE-001.md](REC-20260111-ARM-REFINE-001.md)

---

## 0. 目的

「初期アームプール」「知識グラフ（KG）」「対象トリプル」「追加トリプル候補」が与えられたときに、arm（ルール組）を単位として反復精錬（選択→取得→評価→履歴更新→KG更新）を実行できるコードを実装する。

本計画では、**仮説（hypothesis: ターゲット関係 $r_*$ の候補トリプル）は store-only**（保留ストアへ保存し、反復中はKGへ確定追加しない）を前提とする。

## 1. 入力・出力の最小仕様

### 1.1 入力（与えられるもの）
- 初期アームプール
  - `initial_arms.pkl` または `initial_arms.json`（[build_initial_arms.py](../../build_initial_arms.py) の出力）
  - armは `Arm(arm_type, rule_keys, metadata)` として解釈する
- ルールプール（必須）
  - `initial_rule_pool.pkl`（AmieRules）
  - `rule_keys` を実際の `AmieRule` に解決するために使用
- 初期KG
  - 形式A: triplesディレクトリ（例: `train.txt`）
  - 形式B: 既にロード済みの triple list
- 対象トリプル
  - `target_triples.txt`（主に $(x,r_*,y)$）
- 追加候補トリプル
  - 検証段階: `train_removed.txt` を主（必要なら `train.txt` も合流）
  - 将来: Web/LLM（provenance付き）
- 反復設定
  - `base_output_path`（外部から指定）
  - `n_iter`, `k_sel`, `n_targets_per_arm`, `max_witness_per_head`, selector戦略（`ucb|epsilon_greedy|llm_policy|random`）

### 1.2 出力（反復ごとの成果物）
ディレクトリ規約は `base_output_path/iter_k/`（[simple_active_refine/io_utils.py](../../simple_active_refine/io_utils.py) の `get_iteration_dir` を使用）。

各 `iter_k/` に最低限:
- `selected_arms.json`
- `accepted_evidence_triples.tsv`（KGへ追加された evidence/body トリプル）
- `pending_hypothesis_triples.tsv`（store-only）
- `arm_history.pkl` / `arm_history.json`
- `diagnostics.json`（witness合計、conflict数など）

## 2. 実装方針（コード構成）

既に実装済みの部品を最大限使い、足りないものを追加する。

### 2.1 既にある部品（再利用）
- arm ID: [simple_active_refine/arm.py](../../simple_active_refine/arm.py) `ArmWithId.create()`
- arm 履歴: [simple_active_refine/arm_history.py](../../simple_active_refine/arm_history.py)
- arm 選択: [simple_active_refine/arm_selector.py](../../simple_active_refine/arm_selector.py)
- 反復出力パス: [simple_active_refine/io_utils.py](../../simple_active_refine/io_utils.py) `get_iteration_dir`
- body照合/索引: [simple_active_refine/triples_editor.py](../../simple_active_refine/triples_editor.py)
  - `TripleIndex`, `find_body_triples_for_head`, `count_witnesses_for_head`

### 2.2 新規実装（今回）
- `simple_active_refine/arm_pipeline.py`
  - `ArmDrivenKGRefinementPipeline`（arm版の薄いオーケストレーション）
- `simple_active_refine/arm_triple_acquirer_impl.py`
  - `LocalArmTripleAcquirer`（候補プール(train_removed等)から evidence/body を抽出）
- `simple_active_refine/arm_triple_evaluator_impl.py`
  - `WitnessConflictArmEvaluator`（proxy: witness + conflict）
- `run_arm_refinement.py`
  - CLI entry: 入力パスと設定を受け取り反復を回す

## 3. 主要I/F（データ構造）

### 3.1 ArmAcquisitionResult（arm取得結果）
最低限（v1）:
- `evidence_by_arm: Dict[arm_id, List[Triple]]`
- `witness_by_arm_and_target: Dict[arm_id, Dict[target_triple, int]]`（または集約値）
- `provenance_by_triple: Dict[Triple, Dict]`（localは file/source 程度）

注: evidence と hypothesis を混ぜない。

### 3.2 ArmEvaluationResult（評価結果）
最低限（v1）:
- `accepted_evidence_triples: List[Triple]`
- `pending_hypothesis_triples: List[Triple]`（store-only）
- `arm_rewards: Dict[arm_id, float]`
- `diagnostics: Dict[str, float]`（conflict数、witness合計など）

### 3.3 ArmHistory への記録
各 iteration / arm に対して
- `reward`
- `added_triples`（accepted evidence）
- `target_triples`（そのarmで評価したターゲット集合）
を `ArmEvaluationRecord` として保存。

## 4. 反復アルゴリズム（v1: local候補のみ）

1. 入力読み込み
   - KG: `train.txt` をロードして list/set で保持
   - targets: `target_triples.txt`
   - candidate pool: `train_removed.txt`（+必要なら `train.txt`）
   - arms: `initial_arms.*` → `ArmWithId.create()` で `arm_id` 付与
   - rule pool: `initial_rule_pool.pkl` をロードし、`str(rule)` → `AmieRule` のマップを作る

2. 反復 k=1..N
   - `ArmSelector.select_arms(arm_pool, k_sel, iteration=k)` で選択
   - `LocalArmTripleAcquirer.acquire(...)`
     - iteration内で `TripleIndex(kg + candidate_pool)` を1回だけ作る
     - 各 arm の各ルールに対して、targets を最大 `n_targets_per_arm` サンプル
     - `find_body_triples_for_head` で body を満たす evidence を抽出
     - witness は `count_witnesses_for_head(..., max_witness_per_head)` で集約
   - `WitnessConflictArmEvaluator.evaluate(...)`
     - evidence を受理候補とする（既存KG重複は除外）
     - hypothesis（`r_*`）は、生成しても **pending（store-only）** に入れて KG には加えない
     - arm reward を witness と追加件数で算定（conflictペナルティは将来拡張）
   - KG update
     - `accepted_evidence_triples` のみ KG に追加
   - 出力
     - `iter_k/` に選択arm、受理evidence、pending、履歴、diagnostics を保存

## 5. 重要な設計決定（v1）

- store-only:
  - ターゲット関係 $r_*$ のトリプルは反復中のKGに入れない（pendingへ）
- 出力:
  - 実験ルートは `base_output_path` を外部引数で指定し、`iter_k` は固定
- LLM:
  - selectorは `llm_policy` を実装済み（ただし実験ではAPIキーが必要）
  - v1の取得・評価は local のみで成立させる

## 6. テスト計画（v1）

- unit
  - `tests/test_arm_triple_acquirer.py`: toy triplesで evidence 抽出が行える
  - `tests/test_arm_triple_evaluator.py`: 重複排除・reward算定・store-only動作
  - `tests/test_arm_pipeline_smoke.py`: 1 iteration の最小実行

- integration（任意）
  - `experiments/test_data_for_nationality_v3/` を用いた小規模実行（時間が許す範囲）

## 7. CLI案（run_arm_refinement.py）

必須引数案:
- `--base_output_path`（例: `./experiments/20260111/v3_arm_run`）
- `--initial_arms`（`initial_arms.json` もしくは `initial_arms.pkl`）
- `--rule_pool_pkl`（`initial_rule_pool.pkl`）
- `--dir_triples`（train/valid/test を含む）
- `--target_triples`
- `--candidate_triples`（例: train_removed.txt）

代表コマンド例:
```
python3 run_arm_refinement.py \
  --base_output_path ./experiments/20260111/v3_arm_run \
  --initial_arms ./experiments/20251216/v3_rule/iter_0/initial_arms.pkl \
  --rule_pool_pkl ./experiments/20251216/v3_rule/iter_0/initial_rule_pool.pkl \
  --dir_triples ./experiments/test_data_for_nationality_v3 \
  --target_triples ./experiments/test_data_for_nationality_v3/target_triples.txt \
  --candidate_triples ./experiments/test_data_for_nationality_v3/train_removed.txt \
  --n_iter 3 \
  --k_sel 2 \
  --selector_strategy ucb
```

## 8. 以降の拡張（v2以降）

- WebArmTripleAcquirer: provenance・キャッシュ・レート制御
- conflict LLM adjudication: pending hypothesis の最終判定
- arm拡張: 反復ログから新規ペアarm生成

---

以上を v1 として実装し、まず「与えられた入力から反復精錬を回して evidence を足していく」最小動作を成立させる。
