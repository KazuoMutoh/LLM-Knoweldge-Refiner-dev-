# REC-20260111-ARM_REFINE-003: 実データ（test_data_for_nationality_v3）での1イテレーション検証

作成日: 2026-01-11
更新日: 2026-01-11
参照:
- [docs/records/REC-20260111-ARM_REFINE-002.md](REC-20260111-ARM_REFINE-002.md)
- [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)

---

## 0. 目的

`run_arm_refinement.py` を実データ（nationality）で **1イテレーションだけ**実行し、
- 選択→local取得→proxy評価→履歴保存→evidenceのみKG追加
- hypothesis（ターゲット関係）は store-only（pendingへ）

が成立していることを確認する。

## 1. 検証条件

### 1.1 入力
- 初期アーム（build_initial_arms.py 出力）
  - `/app/experiments/20251216/v3_rule/iter_0/arms/initial_arms.pkl`
- ルールプール（rule_keys解決に使用）
  - `/app/experiments/20251216/v3_rule/iter_0/rules/initial_rule_pool.pkl`
- KG/候補/ターゲット
  - `dir_triples`: `/app/experiments/test_data_for_nationality_v3`
  - `target_triples`: `/app/experiments/test_data_for_nationality_v3/target_triples.txt`
  - `candidate_triples`: `/app/experiments/test_data_for_nationality_v3/train_removed.txt`

### 1.2 実行コマンド

```
python3 run_arm_refinement.py \
  --base_output_path /app/experiments/20260111/v3_arm_run_nationality_v3_1iter_v2 \
  --initial_arms /app/experiments/20251216/v3_rule/iter_0/arms/initial_arms.pkl \
  --rule_pool_pkl /app/experiments/20251216/v3_rule/iter_0/rules/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_nationality_v3 \
  --target_triples /app/experiments/test_data_for_nationality_v3/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_nationality_v3/train_removed.txt \
  --n_iter 1 \
  --k_sel 2 \
  --n_targets_per_arm 50 \
  --max_witness_per_head 50 \
  --selector_strategy ucb
```

### 1.2.1 引数の説明（実装を知らない人向け）

- `--base_output_path`:
  - 出力先のベースディレクトリ。
  - 実行すると `base_output_path/iter_1/` が作られ、`selected_arms.json` や `accepted_evidence_triples.tsv` 等が保存される。
- `--initial_arms`:
  - 初期アームプール。`build_initial_arms.py` が生成する `initial_arms.pkl` または `initial_arms.json`。
  - 中身は `Arm(arm_type, rule_keys, metadata)` のリストで、実行時に安定ID（arm_id）が付与される。
- `--rule_pool_pkl`:
  - ルールプール（`AmieRules`）。
  - `initial_arms` 内の `rule_keys`（文字列化されたルール）を、実際の `AmieRule` に解決するために必須。
- `--dir_triples`:
  - 初期KGのディレクトリ。最低限 `train.txt` を含む。
  - v1では `train.txt` を「現KG」として使い、追加するのは evidence/body トリプルのみ。
- `--target_triples`:
  - 対象トリプルのTSV（今回の例だと nationality）。
  - **store-only（hypothesis）** の判定にも利用（targetに現れるpredicate集合をpending対象とする）。
- `--candidate_triples`:
  - body照合に使う候補トリプル集合（今回は `train_removed.txt`）。
  - v1では基本的に、追加される evidence はここから抽出される。
- `--n_iter`:
  - 反復回数。今回は1なので `iter_1` のみ生成。
- `--k_sel`:
  - 1イテレーションで選択する arm の数（上限制御のひとつ）。
- `--n_targets_per_arm`:
  - 各armが評価に使う target の最大サンプル数（上限制御のひとつ）。
  - 実際は `min(n_targets_per_arm, len(target_triples))`。
- `--max_witness_per_head`:
  - witness（bodyが満たされる置換数）のカウント上限。
  - ※ evidence抽出の件数上限ではなく、witness計測の打ち切り。
- `--selector_strategy`:
  - arm選択戦略。今回は `ucb`（初回は未試行扱いで上位から選択されやすい）。

### 1.3 設定（重要）
- `n_iter=1`（1イテレーション）
- `k_sel=2`（UCBで2アーム選択）
- `n_targets_per_arm=50`（各armで最大50ターゲットをサンプル）
- `max_witness_per_head=50`（witnessカウントの上限）
- store-only判定: **target_triplesに現れるpredicate集合（今回は`/people/person/nationality`）をhypothesisとしてpendingへ**

## 2. 結果（iter_1）

出力ディレクトリ:
- `/app/experiments/20260111/v3_arm_run_nationality_v3_1iter_v2/iter_1`

### 2.1 選択アーム
`selected_arms.json`より（抜粋）:
- `arm_6af063aed93b`（singleton）
  - body: actor-film → film-country ⇒ nationality
  - reward: 106.0
- `arm_b9672dabe85d`（singleton）
  - body: award_nominee(e)→a AND e has nationality b ⇒ a has nationality b
  - reward: 172.0

### 2.2 集計（diagnostics.json）
- `witness_total`: 122.0
- `accepted_evidence_total`: 156.0
- `pending_hypothesis_total`: 1（store-only の仮説トリプル数。実体は `pending_hypothesis_triples.tsv` の行数で、witness>0 かつ現KG未登録で、かつターゲットpredicate（今回は `/people/person/nationality`）に一致するターゲットトリプルを pending に出力した件数）
- `conflict_count`: 0.0（v1では未実装）

### 2.3 生成物
- `accepted_evidence_triples.tsv`: 156行（KGへ追加されたevidence/bodyトリプル）
- `pending_hypothesis_triples.tsv`: 1行（store-only）
  - 例: `/m/0137hn\t/people/person/nationality\t/m/02jx1`
- `arm_history.json` / `arm_history.pkl`: 各armのターゲットサンプル、追加evidence数、reward等を保存

## 3. 所見

- `build_initial_arms.py` の出力（initial_arms.pkl）を初期アームプールとして読み込み、実データで1イテレーション実行できた。
- evidence追加のみKG更新され、ターゲット関係（nationality）は store-only として pending に出力された。
- pendingが1件と少ないのは、今回のサンプル（50×2）で witness>0 かつKG未登録のターゲットが限定的だったため（設計上は正常）。

## 4. 次のアクション（任意）

- `k_sel` や `n_targets_per_arm` を増やして pending の出現頻度を確認する。
- `pending_hypothesis_triples.tsv` の最終確定（adjudication）を別ステップで実装する（v2以降）。
