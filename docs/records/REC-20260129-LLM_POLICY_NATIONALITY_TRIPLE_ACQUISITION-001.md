# REC-20260129-LLM_POLICY_NATIONALITY_TRIPLE_ACQUISITION-001: nationality（head_incident_v1）で model_before を固定し、LLM-policy を適用して「再学習なし」で追加トリプルを取得する実験計画

- 作成日: 2026-01-29
- 最終更新日: 2026-02-01
- 参照:
  - 比較結果（rerun1）: [REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001](REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md)
  - LLM-policy selector 仕様: [RULE-20260117-ARM_SELECTOR_IMPL-001](../rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md)
  - LLM-policy prompt（semantic grounding）実装計画: [REC-20260128-LLM_SELECTOR_SEMANTIC_PROMPT-001](REC-20260128-LLM_SELECTOR_SEMANTIC_PROMPT-001.md)
  - 追加トリプルの再学習評価（iter3/final vs random）計画: [REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001](REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001.md)

## 目的

rerun1 の nationality 実験で用いた **model_before（KG-FIT PairRE）** と、そこで作成済みの **rule_pool / arms** を流用し、
**arm選択のみを LLM-policy に置換**して、
- どの arm が選ばれるか
- どのような evidence/incident triples が追加されるか

を **再学習（KGE retrain）なし**で観測する。

狙い:
- これまでの UCB は proxy 最適化（witness/coverage）に寄りがちだったため、LLM-policy（semantic grounding + entity2text/relation2text）で arm 選択が変わるかを見る。
- 追加トリプル自体はローカル候補集合（train_removed）から取得するため、LLM-policy は「取得元」ではなく「選択方針」の差分として解釈できる。

## 実験条件（固定するもの）

### データセット

- dataset_dir（train/valid/test + text）:
  - `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit`
- target_triples:
  - `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt`
- candidate_triples（local acquisition 用）:
  - `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit/train_removed.txt`

### model_before（固定）

- `/app/models/20260125_rerun1/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100`

注: 今回は **再学習しない**ため、model_before を用いた before/after 評価（Hits/MRR/target_score）は実施しない。
（必要なら、後日 `retrain_and_evaluate_after_arm_run.py` を別途回して評価を追加できる。）

### rule_pool / arms（rerun1 nationality の UCB run から流用）

- rule_pool_pkl:
  - `/app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/rule_pool/initial_rule_pool.pkl`
- initial_arms.json:
  - `/app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arms/initial_arms.json`

意図:
- LLM-policy の効果を「arm選択の差」だけに閉じる（rule pool / arm pool の差を排除）。

## 実験変数（変えるもの）

- selector_strategy: `llm_policy`
- 候補取得: `candidate_source=local`（`train_removed.txt` から）

補足:
- LLM-policy の API 呼び出しは **arm選択**のために実行される（iterationごとに1回程度）。
- evidence の獲得はローカルで完結（Web retrieval は使わない）。

## 実行手順

### 0) 事前チェック（必須）

- `OPENAI_API_KEY` が有効であること（`settings.py` または環境変数）。
- コスト/レート制限を考慮し、まずは **n_iter=1** の smoke test を推奨。

### 1) smoke test（n_iter=1）

出力先（新規）:
- `/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run_smoke/`

コマンド:
- `python run_arm_refinement.py \
  --base_output_path /app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run_smoke \
  --initial_arms /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arms/initial_arms.json \
  --rule_pool_pkl /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/rule_pool/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/train_removed.txt \
  --n_iter 1 --k_sel 3 --n_targets_per_arm 50 \
  --selector_strategy llm_policy \
  --disable_relation_priors`

smoke test の確認観点:
- `iter_1/selected_arms.json` の `policy_text` が空でない
- `iter_1/selected_arms.json` に、semantic grounding（スコアや根拠）に基づく説明が含まれる
- `iter_1/accepted_added_triples.tsv` が生成され、件数が 0 ではない（0でも異常ではないが、まずは設定確認）

### 2) 本番（n_iter=25; rerun1 と同一回数）

出力先（新規）:
- `/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run/`

コマンド:
- `python run_arm_refinement.py \
  --base_output_path /app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run \
  --initial_arms /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arms/initial_arms.json \
  --rule_pool_pkl /app/experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/rule_pool/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/train_removed.txt \
  --n_iter 25 --k_sel 3 --n_targets_per_arm 50 \
  --selector_strategy llm_policy \
  --disable_relation_priors`

## 期待される成果物（最低限）

`/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run/iter_k/` 配下:
- `selected_arms.json`（LLM-policy の reasoning/policy_text/arm rationale を含む）
- `accepted_evidence_triples.tsv`
- `accepted_incident_triples.tsv`（incident on/off に依存）
- `accepted_added_triples.tsv`
- `arm_history.pkl` / `arm_history.json`
- `diagnostics.json`

## 成功条件（受け入れ基準）

- 実行が完走し、各 iteration の `selected_arms.json` に `policy_text` が出力される
- `selected_arms.json` の説明が、target predicate（nationality）と arm body predicates の意味整合に言及する
- 追加トリプル（`accepted_added_triples.tsv`）が取得でき、UCB/Random と比較可能なログが揃う

## 追加の観察（再学習なしでできる範囲）

- arm 選択頻度の比較（LLM-policy vs UCB）
  - 例: `iter_*/selected_arms.json` を集約して、arm_id の出現回数をカウント
- predicate 分布の比較
  - `accepted_added_triples.tsv` を結合して predicate 頻度を集計し、rerun1 の UCB/Random と比べる
- 「意味整合スコア」と proxy 指標の関係
  - `policy_text`（または rationale）にある semantic alignment と `diagnostics.json`（witness/coverage）を並べて傾向を見る

## リスクと対策

- LLMの非決定性:
  - smoke test で挙動を確認後に本番。
  - 必要なら `LLMPolicyArmSelector` の `temperature` を 0 に下げる改修（別チケット）を検討。
- API制限/失敗:
  - iteration 途中で落ちた場合でも、既存の `iter_k` 出力は残るため、原因を特定して再実行（`--base_output_path` を変えるか、壊れた iter を退避）。

## 次の一手（任意）

LLM-policy の選択差が確認できたら、同じ run_dir の `accepted_added_triples.tsv` を用いて
`retrain_and_evaluate_after_arm_run.py` を **別途**回し、before/after の target rescoring まで繋げる。
（ただし本RECのスコープ外：今回は「取得してみる」ことに限定。）

---

## 実験結果（2026-01-29 実施）

### 実行情報

- 実行出力（本番・完走）:
  - `/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run_retry_bg_20260129_202535/`
- selector_strategy: `llm_policy`（再学習なし）
- n_iter=25, k_sel=3, n_targets_per_arm=50
- 集計ファイル:
  - `/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run_retry_bg_20260129_202535/analysis_summary.json`

### 追加トリプル数（accepted_added_triples）

- 総追加数: 887
- iteration別の追加数: min=0 / median=29 / max=87 / mean=35.48
- 0件だったiteration: 2回（25回中）

### arm選択（selected_arms）

- `selected_arms.json` 内の rationale 記録: with=0 / without=75（= 25iter×3arms）
  - 補足: `run.log` には一部iterationで自然言語の理由が出力されているが、JSON側には格納されていない。

上位arm（出現回数）:

- arm_ae6ee4488c19: 13
- arm_8fc0f2710e9e: 12
- arm_268fbd2e5a65: 11
- arm_9aee1a60dfa2: 9
- arm_711c97af9f2c: 7

上位armの内容（代表ルールと直感的意味）:

- arm_ae6ee4488c19（film-country 経由）
  - body predicates: `/film/actor/film./film/performance/film`, `/film/film/country`
  - ルールの形: 「人物?aが映画?fに出演」かつ「映画?fの国が?b」→「?aのnationalityが?b」
  - 直感: “出演作品の国” から国籍を推定するため、妥当性は弱い（ノイズ混入しやすい）。
  - 選ばれやすさ（平均）: reward=74.31 / target_coverage=0.309 / evidence_total=124.38

- arm_8fc0f2710e9e（award nominee からの伝播）
  - body predicates: `/award/award_nominee/award_nominations./award/award_nomination/award_nominee`, `/people/person/nationality`
  - ルールの形: 「あるエンティティ?eが人物?aをaward nomineeとして持つ」かつ「?eのnationalityが?b」→「?aのnationalityが?b」
  - 直感: award周辺のエンティティのnationalityが（データ上）埋まっていると、“同じ賞に紐づく人物へ伝播” できるが、意味的妥当性は状況依存。
  - 選ばれやすさ（平均）: reward=94.92 / target_coverage=0.340 / evidence_total=157.17

- arm_268fbd2e5a65（居住地 × contains）
  - body predicates: `/location/location/contains`, `/people/person/places_lived./people/place_lived/location`
  - ルールの形: 「?aが居住した場所が?f」かつ「?bが?fをcontains」→「?aのnationalityが?b」
  - 直感: 地理的には自然だが、`contains` が “国が都市/地域を含む” になっている必要があり、KG側のスキーマ依存が強い。
  - 選ばれやすさ（平均）: reward=31.55 / target_coverage=0.420 / evidence_total=49.36

- arm_9aee1a60dfa2（award nominee × nationality）
  - body predicates: `/award/award_nominee/award_nominations./award/award_nomination/award_nominee`, `/people/person/nationality`
  - ルールの形: 「人物?aがaward nomineeとして何か?fに紐づく」かつ「?fのnationalityが?b」→「?aのnationalityが?b」
  - 直感: nationality が既知の award関連エンティティ（?f）が多いほど強く働き、結果として award 系トリプルが大量に追加されやすい。
  - 選ばれやすさ（平均）: reward=97.67 / target_coverage=0.351 / evidence_total=153.11

- arm_711c97af9f2c（出生地 × contains）
  - body predicates: `/location/location/contains`, `/people/person/place_of_birth`
  - ルールの形: 「?aの出生地が?f」かつ「?bが?fをcontains」→「?aのnationalityが?b」
  - 直感: 地理的には自然で、place_of_birth 系の証拠に寄る。ただし `contains` の向き/粒度次第で外れも増える。
  - 選ばれやすさ（平均）: reward=25.43 / target_coverage=0.417 / evidence_total=38.14

補足:
- 上位5つのうち3つが award/film 系のため、今回の追加トリプルが award/film predicate に偏った主因になっている。

### predicate分布（accepted_added_triplesのrelation）

上位predicate（件数、上位5）:

- `/award/award_nominee/award_nominations./award/award_nomination/award_nominee`: 372
- `/award/award_winner/awards_won./award/award_honor/award_winner`: 192
- `/film/actor/film./film/performance/film`: 135
- `/award/award_nominee/award_nominations./award/award_nomination/nominated_for`: 75
- `/people/person/places_lived./people/place_lived/location`: 65

target predicate の追加:

- `/people/person/nationality`: 0

関連しそうなpredicate（参考）:

- `/people/person/place_of_birth`: 48

## 考察（目的に対してどうだったか）

### 目的（再掲）

本実験の目的は、「rerun1 で固定した model_before・rule_pool・arms を用いつつ、arm選択のみ LLM-policy に置換したとき、どの arm が選ばれ、どんなトリプルが追加されるか」を **再学習なし**で観測することだった。

### 観測された振る舞い

- 実行は25iter完走し、追加トリプルも十分な量（887件）が得られたため、運用上は LLM-policy の適用が成立した。
- 一方で、追加トリプルの predicate 分布は award / film 系が支配的で、target predicate `/people/person/nationality` は 0件だった。
  - nationality を直接追加しない設計（evidence-first / store-only hypothesis）であることを踏まえても、「nationality に寄与しやすい location/birth 系が主流になる」期待に対し、結果は大きく乖離している。

### 解釈と含意

- arm選択が少数armに集中しており（上位5 arm で 52/75）、探索というより「報酬・coverage の良かった arm へ寄る」挙動が強い可能性がある。
- LLM-policy の説明可能性については、`selected_arms.json` に rationale が残らないため、事後解析の再現性/検証性が弱い。
- 以上より、今回の run は「LLM-policyで回してトリプルを取得する」ことには成功したが、**nationality を改善するための有効な追加知識になっているか**は、この結果だけでは支持できない。
  - 最低限、(1)KGEの再学習ありの定量評価、または (2)追加トリプルのフィルタリング/制約強化が必要。

### 改善案（次の検証）

- semantic grounding の制約を強める
  - arm候補を「location/birth/country 等の語彙・predicate」を含むものに限定（または加点/減点）
  - semantic alignment の閾値を引き上げ、reward 指標だけで選ばれないようにする
- 出力の構造化（rationale）を必ず保存する
  - `run.log` だけでなく `selected_arms.json` に根拠テキストを確実に残す（後段の分析に必須）
- 本実験の目的に沿った最終評価を追加する
  - `retrain_and_evaluate_after_arm_run.py` により、before/after の target rescoring と Hits/MRR を測る

## 結論（観測→含意→次アクション）

### 観測

- LLM-policy を用いた arm-run は、再学習なしでも 25iter 完走し、887件の追加トリプルを取得できた。
- 追加された predicate は award / film 系に大きく偏り、`/people/person/nationality` の追加は 0件だった。
- `selected_arms.json` に rationale が残らず（75/75で欠落）、事後解析の説明可能性が弱い。

### 含意

- 現状の LLM-policy（semantic grounding）だけでは、nationality に寄与しやすい証拠（地理系など）へ十分に誘導できず、proxy 指標（reward/coverage）側に引っ張られている可能性がある。
- rationale がJSONに残らないため、arm選択の妥当性（なぜそのarmが選ばれたか）を定量/定性に追跡しにくい。

### 次アクション

- `retrain_and_evaluate_after_arm_run.py` による再学習ありの定量評価を実施（`exclude_predicate=/people/person/nationality`、ep=2 sanity → ep=100 本番）。
- semantic grounding 制約を強化（location/birth/country への加点/制約、alignment 閾値調整）し、award/film 偏りを抑える。
- LLM-policy の根拠テキストを `selected_arms.json` に保存する（監査可能性・再現性の確保）。

## 追加検証: 地理armの「候補枯渇」仮説（2026-01-29 追記）

ユーザ仮説:
「地理的な arm が（後半で）弱いのは、初期 iteration で `train_removed.txt` 内の地理系候補トリプルを使い切った（枯渇した）ためでは？」

### 1) 実際の追加推移（結果）

集計: `/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run_retry_bg_20260129_202535/geo_depletion_hypothesis_check.json`

- 地理armの選択回数: 42/75（=25iter×3）
- 追加トリプル中の地理predicate: 113/887
- iteration別の地理追加は前半に極端に集中:
  - iter1=50, iter2=36, iter3=12 で 98/113（86.7%）
  - 80%到達は iter3（iter_at_80pct_geo_additions=3）

### 2) `train_removed` 側の残量（使い切り率）

集計: `/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run_retry_bg_20260129_202535/geo_depletion_train_removed_check.json`

- `train_removed.txt` 内の地理候補（全体）: 140
- うち target head（100件）に紐づく地理候補: 140（=地理候補は全て target head 側に存在）
- 実際に追加された地理トリプル: 113
- 地理候補の使用率: 113/140 = 0.807（80.7%）
  - 残候補: 27

predicate別内訳（target head 上での候補 vs 実追加）:

- `/people/person/places_lived./people/place_lived/location`: 候補85 → 追加65（残20）
- `/people/person/place_of_birth`: 候補55 → 追加48（残7）

### 結論（仮説への回答）

- **「地理候補の枯渇（または枯渇に近い状態）」は強く支持される**。
  - 地理追加が iter1-3 に集中し、その時点で候補の大半（80%超）を消費している。
- ただし **完全な枯渇ではなく、`train_removed` 上は 27件の地理候補が残っている**。
  - 後半で地理が伸びない理由には、残候補が (a)既存KGとの重複で弾かれる、(b)unification/マッチングで拾われにくい、(c)選択はされても evidence_new が増えない、等の要因も併存しうる。

---

## 追加検証: 追加トリプルを同数randomと比較してKGE再学習（ep=100）（2026-01-30 追記）

本RECの結論は「再学習なしでは nationality 改善に資するとは言えない」だったため、追加トリプルを train に加えた上で **KGE を再学習**し、
target triple score と KGE metrics（Hits/MRR）がどう変わるかを評価した。

### 実行情報

- 実験出力（単発iteration版・参考）: `/app/experiments/20260129_iter3_final_kge_eval_nationality/`
- 注意（重要）: 上記の単発iteration版は、
  - iter3 actual = `source_arm_run/iter_3/accepted_added_triples.tsv` **単体（=73件）**
  - final actual = `source_arm_run/iter_{final_iter}/accepted_added_triples.tsv` **単体（=9件）**
  を比較しており、意図していた「iter1..3の累積」「iter1..finalの累積」と定義が異なる。
- 本来の意図（累積union版）: iter1..3 / iter1..final の和集合で評価する（下の追補参照）。
- before model（固定）: `/app/models/20260125_rerun1/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100`
- embedding: KG-FIT PairRE（`/app/config_embeddings_kgfit_pairre_fb15k237.json`）
- 評価の比較条件（4条件）:
  - iter3 の追加分（actual） vs 同数random
  - final の追加分（actual） vs 同数random
- 注意: `exclude_predicate=/people/person/nationality` を指定しており、target predicate そのものは train に追加していない（evidence中心の増分のみ）。

### 結果サマリ（ep=100）

共通のbefore（参考）:
- target score (before): -736.4064
- Hits@10 (before): 0.290229
- MRR (before): 0.128665

| 条件 | 追加triple数 | target score Δ | Hits@10 Δ | MRR Δ | summary.json |
|---|---:|---:|---:|---:|---|
| iter3 actual | 73 | +7.7187 | -0.00143 | -0.00159 | `/app/experiments/20260129_iter3_final_kge_eval_nationality/ep100_full/cond1_iter3_actual/retrain_eval/summary.json` |
| iter3 random | 73 | +12.1006 | -0.01052 | -0.00213 | `/app/experiments/20260129_iter3_final_kge_eval_nationality/ep100_full/cond2_iter3_random/retrain_eval/summary.json` |
| final actual | 9 | -3.4659 | -0.00713 | -0.00453 | `/app/experiments/20260129_iter3_final_kge_eval_nationality/ep100_full/cond3_final_actual/retrain_eval/summary.json` |
| final random | 9 | +5.3427 | -0.01091 | -0.00255 | `/app/experiments/20260129_iter3_final_kge_eval_nationality/ep100_full/cond4_final_random/retrain_eval/summary.json` |

### 解釈（要点）

- iter3（73件）では、target score は **randomの方が改善幅が大きい**（+12.10 vs +7.72）。一方で Hits@10/MRR の劣化は actual の方が小さい。
- final（9件）では、target score は **actual が悪化**（-3.47）し、random は改善（+5.34）。Hits@10 の劣化は actual の方が小さいが、MRR は actual の方が悪い。
- 総合すると、今回の ep=100 再学習評価では、LLM-policy run で得た増分（actual）が **同数randomより一貫して優位**とは言えない（特に target score で劣後）。
- final の増分が 9件と少ないため、ここはノイズが大きい可能性が高い（再現性のためには seed を変えた反復が必要）。

### 追補: 累積（iter1..3 / iter1..final の和集合）での再評価（完走: 2026-01-31）

ユーザ指摘の通り、iter3までの累積（例: 地理系だけでもiter1-3で98件）と整合させるには、
**iter1..k の accepted_added_triples を和集合（重複除外）**して train に追加する必要がある。

- 実験出力（累積union版・新規）: `/app/experiments/20260130_iter3_final_kge_eval_nationality_cumulative/`
  - ログ: `/app/experiments/20260130_iter3_final_kge_eval_nationality_cumulative/master_run.log`
- 追加トリプル数（和集合・重複除外）:
  - iter1..3 union: 159（内訳: iter1=50, iter2=36, iter3=73; 重複なし）
  - iter1..final(=iter25) union: 887（重複なし）
- 実行方針: ep=2 パイロット → ep=100 本番を自動実行（途中で確認は挟まない）

完走後に、累積union版の `summary.json`（4条件）をこの節へ追記して、単発iteration版よりもこちらを主結果として扱う。

#### 結果（ep100_full / 累積union版）

# Summary: 20260130_iter3_final_kge_eval_nationality_cumulative / ep100_full

## Before (common)

- target score (before): -736.4064
- Hits@10 (before): 0.290229
- MRR (before): 0.128665

| 条件 | 追加triple数 | target score Δ | Hits@10 Δ | MRR Δ | summary.json |
|---|---:|---:|---:|---:|---|
| cond1_iter3_actual | 159 | +25.5755 | -0.009781 | -0.004122 | `/app/experiments/20260130_iter3_final_kge_eval_nationality_cumulative/ep100_full/cond1_iter3_actual/retrain_eval/summary.json` |
| cond2_iter3_random | 159 | +5.7335 | -0.023494 | -0.008317 | `/app/experiments/20260130_iter3_final_kge_eval_nationality_cumulative/ep100_full/cond2_iter3_random/retrain_eval/summary.json` |
| cond3_final_actual | 887 | +8.1312 | -0.004866 | -0.000538 | `/app/experiments/20260130_iter3_final_kge_eval_nationality_cumulative/ep100_full/cond3_final_actual/retrain_eval/summary.json` |
| cond4_final_random | 887 | +23.0920 | -0.005013 | +0.002285 | `/app/experiments/20260130_iter3_final_kge_eval_nationality_cumulative/ep100_full/cond4_final_random/retrain_eval/summary.json` |

## Comparisons

- iter1..3 union（159件） actual - random: target score Δ=+19.8420, Hits@10 Δ=+0.013713, MRR Δ=+0.004195
- iter1..final union（887件） actual - random: target score Δ=-14.9608, Hits@10 Δ=+0.000147, MRR Δ=-0.002824

- 出力ディレクトリ: `/app/experiments/20260130_iter3_final_kge_eval_nationality_cumulative`
- phase: `ep100_full`

