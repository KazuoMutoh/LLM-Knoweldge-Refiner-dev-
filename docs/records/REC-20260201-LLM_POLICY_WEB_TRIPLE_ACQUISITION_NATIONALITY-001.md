# REC-20260201-LLM_POLICY_WEB_TRIPLE_ACQUISITION_NATIONALITY-001: nationality（head_incident_v1）で初期KGE/armsを固定し、arm選択=LLM-policy + candidate_source=web でWebからトリプル取得（iter=25）

- 作成日: 2026-02-01
- 最終更新日: 2026-02-01
- 参照:
  - LLM-policy（arm選択）+ local候補での25iter実験と所見: [REC-20260129-LLM_POLICY_NATIONALITY_TRIPLE_ACQUISITION-001](REC-20260129-LLM_POLICY_NATIONALITY_TRIPLE_ACQUISITION-001.md)
  - Web retrieval（entity/triple取得）標準: [RULE-20260124-WEB_ENTITY_RETRIEVAL-001](../rules/RULE-20260124-WEB_ENTITY_RETRIEVAL-001.md)
  - arm pipeline（candidate_source=local|web）仕様: [RULE-20260117-ARM_PIPELINE_IMPL-001](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)
  - LLM-policy selector 仕様: [RULE-20260117-ARM_SELECTOR_IMPL-001](../rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md)
  - （参考）nationalityでの最小web試行: [REC-20260124-NATIONALITY_WEB_TRIAL-001](REC-20260124-NATIONALITY_WEB_TRIAL-001.md)

## 目的

REC-20260129 では、arm選択=LLM-policy に置換した上で candidate_source=local（train_removed）で25iterを回し、追加トリプルは得られた一方で award/film 系 predicate に偏った。
また本パイプラインは store-only hypothesis の設計のため、target predicate（`/people/person/nationality`）そのものは追加対象ではなく、**「targetに意味的に価値のある証拠（evidence）トリプルを追加することで、再学習後に target triple スコアを上げる」**ことを狙う。

さらに、REC-20260129 の「追補: 累積（iter1..k の和集合）での再評価」では、nationalityに対して意味的に価値のある（地理系）トリプルを追加した場合は target triple のスコアが向上し、価値の薄いトリプルを追加した場合はスコアが低下する、という傾向が観測された。
一方で、candidate_source=local のときは train_removed に含まれる候補数が有限であり、**改善幅（= 追加できる有益トリプルの上限）も限定される**。

本実験では、
- **初期KGE（model_before）/ rule_pool / arms を固定**しつつ
- **candidate_source=web** に切り替え、Webから候補トリプルを取得
- **arm選択は LLM-policy のまま**

として、
- train_removed の制限を超えて、**nationalityに寄与しやすい証拠（地理系など）をより多く獲得できるか**
- その結果として、**再学習後の target triple スコアを（local候補より）大きく向上できるか**
- Web由来のノイズ/新規entity導入がどれくらい発生するか

を、**iter=25** で観測する。

※目的が「スコア改善」であるため、本RECでは arm-run（取得）だけでなく、**累積（iter1..final の和集合）でのKGE再学習評価までを必須**とする。

## 実験条件（固定するもの）

### データセット

- dataset_dir（train/valid/test + text）:
  - `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit`
- target_triples:
  - `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt`
- candidate_triples（CLI必須のためのダミー。**train_removedは使わない**）:
  - `/app/tmp/debug/empty_candidate_triples.tsv`

補足:
- candidate_source=web の場合も `run_arm_refinement.py` のI/F上 `--candidate_triples` は必須だが、本実験では **train_removedを使わない**。
- `--candidate_triples` には空ファイル（ダミー）を渡し、`--disable_incident_triples` を指定して incident triple augmentation も無効化する（= ローカル候補に依存しない）。

### 初期KGE（model_before）

本実験の比較軸を揃えるため、REC-20260129 と同一の初期KGE（KG-FIT PairRE）を「初期条件」として固定する。

- model_before:
  - `/app/models/20260125_rerun1/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100`

注意:
- `run_arm_refinement.py` は model_before を直接は参照しない（proxy指標で反復する）。
- ただし後段で `retrain_and_evaluate_after_arm_run.py` により before/after を比較する際に、同一の before モデルを使うことで結果が解釈しやすい。

### rule_pool / arms（rerun1 nationality の UCB run から流用）

- rule_pool_pkl:
  - `/app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/rule_pool/initial_rule_pool.pkl`
- initial_arms.json:
  - `/app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arms/initial_arms.json`

意図:
- arm候補集合の差を排除し、**candidate_source=web による取得元の差**と、**LLM-policy による選択の差**に焦点を当てる。

## 実験変数（変えるもの）

- selector_strategy: `llm_policy`
- candidate_source: `web`

Web取得パラメータ（デフォルト値を起点に調整）:
- `web_llm_model`: `gpt-4o`
- `web_max_targets_total_per_iteration`: 20（1iterあたりの (arm,target) クエリ上限）
- `web_max_triples_per_iteration`: 200（1iterあたり保持するweb候補トリプル上限）
- `disable_web_search`: false（web_search_preview を使う）
- `disable_entity_linking`: false（KG既存entityへのリンクを試みる）

## 実行手順

### 0) 事前チェック（必須）

- `OPENAI_API_KEY` が有効（`settings.py` または環境変数）。
- ネットワーク到達性（Web取得が走るため）。
- コスト/レート制限対策として、まず smoke test（n_iter=1）で挙動確認。

### 1) smoke test（n_iter=1）

出力先（新規）:
- `/app/experiments/20260201_llm_policy_web_nationality_head_incident_v1_from_rerun1/arm_run_web_smoke/`

コマンド（コスト制限のため cap を小さくする例）:
- `python run_arm_refinement.py \
  --base_output_path /app/experiments/20260201_llm_policy_web_nationality_head_incident_v1_from_rerun1/arm_run_web_smoke \
  --initial_arms /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arms/initial_arms.json \
  --rule_pool_pkl /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/rule_pool/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
  --candidate_triples /app/tmp/debug/empty_candidate_triples.tsv \
  --n_iter 1 --k_sel 3 --n_targets_per_arm 50 \
  --candidate_source web \
  --web_max_targets_total_per_iteration 5 \
  --web_max_triples_per_iteration 50 \
  --selector_strategy llm_policy \
  --disable_incident_triples \
  --disable_relation_priors`

smoke test の確認観点:
- `iter_1/triple_acquirer_io.json` が生成され、`mode=web_search` で候補トリプルが出力される
- `iter_1/accepted_evidence_triples.tsv` / `accepted_added_triples.tsv` が生成される
- `iter_1/selected_arms.json` が生成される（LLM-policyの出力がログに出ること）

### 2) 本番（n_iter=25）

出力先（新規）:
- `/app/experiments/20260201_llm_policy_web_nationality_head_incident_v1_from_rerun1/arm_run_web_iter25/`

コマンド（まずはデフォルト上限を採用）:
- `python run_arm_refinement.py \
  --base_output_path /app/experiments/20260201_llm_policy_web_nationality_head_incident_v1_from_rerun1/arm_run_web_iter25 \
  --initial_arms /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arms/initial_arms.json \
  --rule_pool_pkl /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/rule_pool/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
  --candidate_triples /app/tmp/debug/empty_candidate_triples.tsv \
  --n_iter 25 --k_sel 3 --n_targets_per_arm 50 \
  --candidate_source web \
  --web_llm_model gpt-4o \
  --web_max_targets_total_per_iteration 20 \
  --web_max_triples_per_iteration 200 \
  --selector_strategy llm_policy \
  --disable_incident_triples \
  --disable_relation_priors`

運用メモ:
- Web取得が不安定な場合は、`--disable_web_search` を付けて「web_search_preview無し（LLMのみ）」へ切替して切り分けする。
- 新規entityの導入が多すぎる場合は、`--disable_entity_linking` のオン/オフ比較や、incident triple augmentation の上限（`--max_incident_candidate_triples_per_iteration`）導入を検討する。

## 期待される成果物（最低限）

`.../arm_run_web_iter25/iter_k/` 配下:
- `selected_arms.json`
- `accepted_evidence_triples.tsv`
- `accepted_incident_triples.tsv`（incident on/off に依存）
- `accepted_added_triples.tsv`
- `triple_acquirer_io.json`（web取得候補のダンプ。`mode=web_search` を含む）
- `arm_history.pkl` / `arm_history.json`
- `diagnostics.json`

## 観察・評価（本RECのスコープ）

### A. 取得量・偏りの定量

- 追加トリプル数（iteration別/総数）: `accepted_added_triples.tsv` の件数
- predicate 分布（上位k、long tail）: `accepted_added_triples.tsv` の relation 集計
- target近傍の兆候:
  - `/people/person/place_of_birth`, `/people/person/places_lived...`, `/location/location/contains`, `/location/.../country` 等の増加

補足:
- store-only hypothesis / web候補フィルタの都合で、`/people/person/nationality` 自体は基本的に追加されない前提（=0件でも正常）。

### B. Web由来ノイズの兆候

- 新規entity数の増加（IDのパターンや、incident triples の増え方）
- 同一事実の重複（subject/predicate/objectの重複率）
- 無関係predicate（award/film系）への偏りが継続していないか

### C. 比較対象

- baseline: REC-20260129（candidate_source=local、同じarms/rule_pool、iter=25）
- （可能なら）REC-20260126 rerun1（UCB/random）との定性的比較

### D. KGE再学習評価（必須）

目的（スコア改善）に整合させるため、arm-runの出力（iter1..final の `accepted_added_triples.tsv` の和集合）を train に追加して再学習し、target triple スコアの before/after を評価する。

評価コマンド（本番run_dirに対して実行）:
- quick sanity（ep=2）:
  - `python retrain_and_evaluate_after_arm_run.py \
    --run_dir /app/experiments/20260201_llm_policy_web_nationality_head_incident_v1_from_rerun1/arm_run_web_iter25 \
    --dataset_dir /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
    --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
    --model_before_dir /app/models/20260125_rerun1/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100 \
    --after_mode retrain \
    --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
    --num_epochs 2 \
    --exclude_predicate /people/person/nationality \
    --force_retrain`
- main（ep=100）:
  - `python retrain_and_evaluate_after_arm_run.py \
    --run_dir /app/experiments/20260201_llm_policy_web_nationality_head_incident_v1_from_rerun1/arm_run_web_iter25 \
    --dataset_dir /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
    --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
    --model_before_dir /app/models/20260125_rerun1/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100 \
    --after_mode retrain \
    --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
    --num_epochs 100 \
    --exclude_predicate /people/person/nationality \
    --force_retrain`

比較のため、baseline（REC-20260129 の local run_dir）にも同じ評価スクリプトを適用し、target score Δ を比較する（追加triple数の違いがある場合は要注意で併記する）。

## 成功条件（受け入れ基準）

- iter=25 が完走し、各iterで `triple_acquirer_io.json` が出力される
- `accepted_added_triples.tsv` が取得でき、local候補のrunと比較可能なログが揃う
- predicate 分布が「award/film一色」から緩和し、location/birth/country 系が相対的に増える（少なくとも上位に出る）

加えて（本目的）:
- `retrain_and_evaluate_after_arm_run.py`（累積union）で、target triple スコアが before から改善（Δ>0）する
- 可能なら、REC-20260129（local候補）での同条件評価よりも target score 改善幅が大きい

## リスクと対策

- Web取得のノイズ/ハルシネーション:
  - `web_max_triples_per_iteration` を下げて「薄く広く」よりも「少数・高確度」へ寄せる（または逆）
  - entity linking 有効化で既存KGへの寄せを強める（ただし誤リンクには注意）
- API制限/失敗:
  - smoke test → 本番の順で段階実行
  - 失敗iterationがあっても `iter_k` 出力は残るため、原因特定して再実行（出力先を変える）
- LLM-policy の説明可能性（rationaleがJSONに残らない問題）:
  - 解析時は `run.log` も併用し、必要なら `selected_arms.json` へのrationale保存を別タスク化する

## 次の一手（任意）

- Web由来トリプルの品質管理（predicate/ソース別フィルタ、entity linking の強化）を導入して、ノイズ混入時の劣化を抑える
- 「地理系（有益）/非地理系（有害）」の切り分け再現のため、追加トリプルを predicate でサブセット化して再学習評価する（別タスク）

---

## 実験結果（追記: KGE再学習 ep=2 sanity、2026-02-01 実施）

### 実行情報

- run_dir:
  - `/app/experiments/20260201_llm_policy_web_nationality_head_incident_v1_from_rerun1/arm_run_web_iter25`
- 再学習+評価の出力:
  - `.../retrain_eval/summary.json`
  - `.../retrain_eval/evaluation/iteration_metrics.json`
  - `.../retrain_eval/updated_triples/{train,valid,test}.txt`
  - `.../retrain_eval/updated_triples/added_triples.tsv`

### 追加トリプル集約（累積union）

- `iter_1..25` の `accepted_added_triples.tsv` の累積union: 87件
- `retrain_eval/updated_triples/added_triples.tsv` の行数: 87

### 指標（ep=2）

`iteration_metrics.json` より:

- target_score_change: +568.0476（-736.4064 → -168.3588）
- Hits@10_change: -0.1323（0.2902 → 0.1579）
- MRR_change: -0.0492（0.1287 → 0.0794）

補足（出力の整合性）:

- `summary.json` の `updated_dataset.n_added_used=0` と、`added_triples.tsv=87行` が矛盾している（要確認）。

## 結論（観測→含意→次アクション）

### 観測

- web run（iter=25）の累積unionは 87件と少量だった。
- ep=2 sanity では、target score は大きく改善した一方で、Hits@10/MRR は大きく悪化した。
- `summary.json` の「追加件数」周りに整合性の崩れが見える（ファイル実体は87件あるが、n_added_used=0 と出る）。

### 含意

- ep=2 の単発結果は不安定になりやすく、改善/悪化の判断には不十分（ep=100 と複数seedの対照が必要）。
- 集計/レポートの件数が不整合だと、後段の比較（local vs web、random対照）で誤解釈リスクがある。

### 次アクション

- `retrain_and_evaluate_after_arm_run.py` の集計（`n_added_used` 等）と `updated_triples/added_triples.tsv` の対応を確認し、レポート出力の整合性を担保する。
- ep=2 sanity を「数値の向き」ではなく「完走とI/O確認」に限定し、問題なければ ep=100 本番（＋random対照）へ進める。
- 取得数（87件）がボトルネックなら、`web_max_targets_total_per_iteration` / `web_max_triples_per_iteration` の見直しや、arm/prompt側で地理系証拠を増やす誘導を検討する。
