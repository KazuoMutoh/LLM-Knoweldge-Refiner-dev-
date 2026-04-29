# REC-20260203-LLM_POLICY_PLACE_OF_BIRTH_CUMULATIVE_KGE_RETRAIN-001: LLM-policy（place_of_birth）の累積追加トリプルでKGEを再学習し、同数Randomと比較する（実験計画）

- 作成日: 2026-02-03
- 最終更新日: 2026-02-03

参照（同条件の先行実験: nationality）:
- [REC-20260201-LLM_POLICY_NATIONALITY_CUMULATIVE_KGE_RETRAIN-001](REC-20260201-LLM_POLICY_NATIONALITY_CUMULATIVE_KGE_RETRAIN-001.md)

参照（仕様）:
- テストデータ生成: [docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md](../rules/RULE-20260119-MAKE_TEST_DATASET-001.md)
- 再学習評価プロトコル: [docs/rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md](../rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md)
- LLM-policy selector 仕様: [docs/rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md](../rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md)
- arm pipeline 仕様: [docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)

## 1. 目的

LLM-policy により選別された追加トリプル集合が、同数のランダム追加よりも KGE（KG-FIT + PairRE）を改善するかを、
**累積union**の評価設定で検証する（対象関係: `/people/person/place_of_birth`）。

nationality と同様に、LLM-policy arm-run の追加集合のうち
- 早期（iter1..3）の累積union
- 最終（iter1..final）の累積union（本計画では n_iter=25 を想定し、iter1..25）

をそれぞれ追加して KGE を再学習し、同数ランダム追加を対照として、target triples のスコアとリンク予測指標（Hits@10, MRR）の差を評価する。

## 2. 実験方法（固定条件）

### 2.1 データセット

- dataset_dir（head_incident_v1 + KG-FIT用テキスト付与済み）:
  - `/app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit`
- target_triples:
  - `/app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/target_triples.txt`
- candidate_triples（local acquisition; train_removed からのみ追加）:
  - `/app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/train_removed.txt`

補足:
- 本計画は「local候補（train_removed）からの追加」を前提とする（candidate_source=local）。
- incident triples の扱いは nationality の LLM-policy local run と合わせ、`run_arm_refinement.py` のデフォルト挙動（disableしない）を踏襲する。

### 2.2 before（初期KGE）

比較可能性のため、rerun1 で作成済みの before モデルを固定する。

- model_before_dir:
  - `/app/models/20260126_rerun1/fb15k237_kgfit_pairre_place_of_birth_head_incident_v1_before_ep100_lowmem_bs128`

### 2.3 rule_pool / arms（rerun1 place_of_birth の UCB run から流用）

- rule_pool_pkl:
  - `/app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/rule_pool/initial_rule_pool.pkl`
- initial_arms.json:
  - `/app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/arms/initial_arms.json`

意図:
- arm候補集合の差を排除し、**arm選択=LLM-policy** による差だけを観測可能にする。

### 2.4 KGE / 評価

- 埋め込みモデル: KG-FIT（PairRE）
- embedding_config（before と同一条件を要求）:
  - `/app/config_embeddings_kgfit_pairre_fb15k237_lowmem_bs128.json`
- エポック数（afterの再学習）:
  - main: 100（before/afterで統一）
  - sanity: 2（更新トリプル生成・学習導線確認用）
- リーク防止:
  - after 学習に追加するトリプルから、target predicate `/people/person/place_of_birth` を除外する
- 評価指標:
  - target score（target triples のスコア集約）
  - Hits@10
  - MRR

## 3. 実験変数（変えるもの）

- selector_strategy: `llm_policy`
- candidate_source: `local`
- relation priors: 無効化（`--disable_relation_priors`）

## 4. 実行手順

### 4.1 LLM-policy arm-run（追加トリプル獲得）

事前条件:
- `OPENAI_API_KEY` が有効であること

出力先（新規）:
- `/app/experiments/20260203_llm_policy_place_of_birth_head_incident_v1_from_rerun1/`

smoke test（n_iter=1）:
- `python run_arm_refinement.py \
  --base_output_path /app/experiments/20260203_llm_policy_place_of_birth_head_incident_v1_from_rerun1/arm_run_smoke \
  --initial_arms /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/arms/initial_arms.json \
  --rule_pool_pkl /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/rule_pool/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/train_removed.txt \
  --n_iter 1 --k_sel 3 --n_targets_per_arm 50 \
  --selector_strategy llm_policy \
  --disable_relation_priors`

本番（n_iter=25）:
- `python run_arm_refinement.py \
  --base_output_path /app/experiments/20260203_llm_policy_place_of_birth_head_incident_v1_from_rerun1/arm_run \
  --initial_arms /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/arms/initial_arms.json \
  --rule_pool_pkl /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/rule_pool/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/train_removed.txt \
  --n_iter 25 --k_sel 3 --n_targets_per_arm 50 \
  --selector_strategy llm_policy \
  --disable_relation_priors`

### 4.2 累積union（iter1..3 / iter1..final）の作成

`retrain_and_evaluate_after_arm_run.py` は `run_dir/iter_*/accepted_added_triples.tsv` を全結合するため、
入力制御は「run_dirを作り分ける」運用で統一する（プロトコル参照）。

- actual（iter1..3 union）:
  - `arm_run` から `iter_1..iter_3` のみをコピーして `arm_run_union_iter1_3/` を作成
- actual（iter1..25 union）:
  - `arm_run` をそのまま使用

### 4.3 同数Random baseline 用 run_dir の作成

Random baseline は「候補集合（train_removed）から、actual と同数N件をサンプル」して run_dir を作る。

重要:
- `/people/person/place_of_birth` を含むトリプルは **サンプル元から除外**する（除外は評価時にも行うが、件数整合のため事前除外を推奨）。
- 重複トリプルは除外（ユニーク化）。

作成するrun_dir（例）:
- `arm_run_random_union_iter1_3_seed0/iter_1/accepted_added_triples.tsv`
- `arm_run_random_union_iter1_25_seed0/iter_1/accepted_added_triples.tsv`

※ Random seed を変えて分散推定したい場合は、seed=0..3 を追加で作る。

## 5. KGE再学習評価（actual / random）

評価は全て `--after_mode retrain` とし、cumulative union の比較を行う。

### 5.1 sanity（ep=2）

- actual（iter1..3）:
  - `python retrain_and_evaluate_after_arm_run.py \
    --run_dir /app/experiments/20260203_llm_policy_place_of_birth_head_incident_v1_from_rerun1/arm_run_union_iter1_3 \
    --dataset_dir /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit \
    --target_triples /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/target_triples.txt \
    --model_before_dir /app/models/20260126_rerun1/fb15k237_kgfit_pairre_place_of_birth_head_incident_v1_before_ep100_lowmem_bs128 \
    --after_mode retrain \
    --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237_lowmem_bs128.json \
    --num_epochs 2 \
    --exclude_predicate /people/person/place_of_birth \
    --force_retrain`

- random（iter1..3, seed0）:
  - `python retrain_and_evaluate_after_arm_run.py \
    --run_dir /app/experiments/20260203_llm_policy_place_of_birth_head_incident_v1_from_rerun1/arm_run_random_union_iter1_3_seed0 \
    --dataset_dir /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit \
    --target_triples /app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/target_triples.txt \
    --model_before_dir /app/models/20260126_rerun1/fb15k237_kgfit_pairre_place_of_birth_head_incident_v1_before_ep100_lowmem_bs128 \
    --after_mode retrain \
    --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237_lowmem_bs128.json \
    --num_epochs 2 \
    --exclude_predicate /people/person/place_of_birth \
    --force_retrain`

同様に iter1..25（actual/random）も実行する。

### 5.2 main（ep=100）

sanity が通った条件について、`--num_epochs 100` で同様に実行する。

## 6. 解析・レポート方針

主要比較:
- actual - random（差分）
  - target score Δ
  - Hits@10 Δ
  - MRR Δ

提示形式:
- before（共通）
- actual-random（iter1..3 union / iter1..25 union）

## 7. 成功条件（受け入れ基準）

- arm-run が完走し、各iterで `iter_k/accepted_added_triples.tsv` と `iter_k/selected_arms.json` が生成される
- retrain_eval が完走し、`run_dir/retrain_eval/summary.json` と `run_dir/retrain_eval/evaluation/iteration_metrics.json` が出力される
- actual と random の比較で、少なくとも iter1..3 union について差分が安定して計測できる（可能なら seed 複数）

## 8. リスクと対策

- LLMの非決定性:
  - まず smoke test を行い、挙動とコストを確認してから本番を回す
- Random baseline の分散:
  - seed を複数（推奨: 4）にして平均±分散を推定する
- 件数整合（exclude_predicateの影響）:
  - randomサンプル時点で target predicate を除外しておく

---

## 付録: 期待されるディレクトリ構造（例）

- `/app/experiments/20260203_llm_policy_place_of_birth_head_incident_v1_from_rerun1/`
  - `arm_run/iter_1..iter_25/...`
  - `arm_run_union_iter1_3/iter_1..iter_3/...`
  - `arm_run_random_union_iter1_3_seed0/iter_1/accepted_added_triples.tsv`
  - `arm_run_random_union_iter1_25_seed0/iter_1/accepted_added_triples.tsv`
  - `*/retrain_eval/summary.json`
