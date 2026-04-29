# REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001: UCB vs Random(seed=0) 3関係 rerun1 統合サマリ＋target_score非常勝の考察

- 作成日: 2026-01-28
- 最終更新日: 2026-01-28
- 参照（実験計画）: [REC-20260123-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001](REC-20260123-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md)
- 実験ベースディレクトリ: `experiments/20260126_rerun1/`

## 目的

incident triples 除去データ（nationality / place_of_birth / profession）に対し、KG-FIT(PairRE)で
- UCB（arm=10, priors off）によりトリプル追加→retrain/eval
- 追加件数を合わせた Random(seed=0) でトリプル追加→retrain/eval

を比較し、(a) summary.json 指標、(b) minmax(train) 正規化の per-target 再スコアリングで勝敗を整理する。
加えて「UCBが target_score で常勝しない」理由を、上昇/下降ターゲットと追加トリプルの意味的・埋め込み的観点から考察する。

## 入力（per-relation 比較レポート）

- nationality: [experiments/20260126_rerun1/compare_ucb_vs_random_seed0_nationality_head_incident_v1_20260126b.md](../../experiments/20260126_rerun1/compare_ucb_vs_random_seed0_nationality_head_incident_v1_20260126b.md)
- place_of_birth: [experiments/20260126_rerun1/compare_ucb_vs_random_seed0_place_of_birth_head_incident_v1_20260127a.md](../../experiments/20260126_rerun1/compare_ucb_vs_random_seed0_place_of_birth_head_incident_v1_20260127a.md)
- profession: [experiments/20260126_rerun1/compare_ucb_vs_random_seed0_profession_head_incident_v1_20260127b.md](../../experiments/20260126_rerun1/compare_ucb_vs_random_seed0_profession_head_incident_v1_20260127b.md)

補助（本ドキュメント作成時の集計JSON）:
- [experiments/20260126_rerun1/analysis_ucb_vs_random_seed0_3relations_20260128.json](../../experiments/20260126_rerun1/analysis_ucb_vs_random_seed0_3relations_20260128.json)

追加トリプルの「タイプ（predicate）」分析（本ドキュメント追記）:
- [experiments/20260126_rerun1/analysis_added_triples_predicates_20260128.md](../../experiments/20260126_rerun1/analysis_added_triples_predicates_20260128.md)

## Summary.json 指標（after retrain/eval）

| relation | n_added_used | target_score_change (UCB / Random / winner) | mrr_change (UCB / Random / winner) | hits@10_change (UCB / Random / winner) |
|---|---:|---|---|---|
| nationality | 445 | 18.4574 / 16.5572 / UCB | -0.00183956 / -0.00413892 / UCB | -0.0149415 / -0.000245749 / Random |
| place_of_birth | 45 | -1.50677 / 2.17452 / Random | -0.000631236 / 0.00318363 / Random | 0.00388016 / 0.00574656 / Random |
| profession | 244 | 2.47249 / 3.29134 / Random | -0.00127502 / -0.00194813 / UCB | 0.0127908 / 0.00838817 / UCB |

## Minmax(train) 正規化ターゲット再スコア（per-target Δの集計）

| relation | Δmean (UCB / Random / winner) | improved_frac (UCB / Random / winner) | n_targets |
|---|---|---|---:|
| nationality | 0.0132431 / 0.00370567 / UCB | 0.692308 / 0.555556 / UCB | 117 |
| place_of_birth | 0.0111781 / 0.0183667 / Random | 0.67 / 0.74 / Random | 100 |
| profession | 0.0396948 / 0.0143782 / UCB | 0.909091 / 0.710744 / UCB | 363 |

## UCB: arm一覧と最終proxy報酬（arm=10）

ここでは UCB で用いた arm（ルール組）と、最終イテレーション時点の proxy 報酬統計を一覧化する。

- arm定義: 各run_dirの `arms/initial_arms.json`
- 報酬ログ: 各run_dirの `arm_run/iter_25/arm_history.pkl`
- 報酬の意味: arm選択（bandit）に使う proxy reward（witness/coverage 等の診断値から算出されるスカラー。詳細は `selected_arms.json` の `reward` / `diagnostics` を参照）

注: 下表の `mean_reward` は「そのarmが評価された回に得た `reward` の平均」、`sum_reward` は合計、`recent_mean` は直近3回平均、`total_added` は当該arm経由で追加されたトリプル総数。

### nationality

- arm定義: [experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arms/initial_arms.json](../../experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arms/initial_arms.json)
- 報酬ログ: [experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arm_run/iter_25/arm_history.pkl](../../experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260126b/arm_run/iter_25/arm_history.pkl)

| arm_id | rule (short) | n_eval | mean_reward | recent_mean | sum_reward | total_added |
|---|---|---:|---:|---:|---:|---:|
| arm_8fc0f2710e9e | /a/award_nominee/award_nominations./a/award_nomination/award_nominee + /p/p/nationality => /p/p/nationality | 23 | 40.913 | 30.667 | 941.0 | 189 |
| arm_ae6ee4488c19 | /f/actor/film./f/performance/film + /f/film/country => /p/p/nationality | 23 | 29.130 | 30.333 | 670.0 | 134 |
| arm_591b05ecf887 | /a/award_nominee/award_nominations./a/award_nomination/nominated_for + /f/film/country => /p/p/nationality | 21 | 18.333 | 15.333 | 385.0 | 78 |
| arm_268fbd2e5a65 | /loc/loc/contains + /p/p/places_lived./people/place_lived/location => /p/p/nationality | 1 | 16.000 | 16.000 | 16.0 | 8 |
| arm_9aee1a60dfa2 | /a/award_nominee/award_nominations./a/award_nomination/award_nominee + /p/p/nationality => /p/p/nationality | 1 | 16.000 | 16.000 | 16.0 | 8 |
| arm_475a56436592 | /a/award_winner/awards_won./a/award_honor/award_winner + /p/p/nationality => /p/p/nationality | 1 | 16.000 | 16.000 | 16.0 | 8 |
| arm_711c97af9f2c | /loc/loc/contains + /p/p/place_of_birth => /p/p/nationality | 2 | 14.000 | 14.000 | 28.0 | 13 |
| arm_6a06bdebf12f | /base/biblioness/bibs_location/country + /p/p/places_lived./people/place_lived/location => /p/p/nationality | 1 | 11.000 | 11.000 | 11.0 | 5 |
| arm_f539a2eebb5a | /a/award_winner/awards_won./a/award_honor/award_winner + /p/p/nationality => /p/p/nationality | 1 | 4.000 | 4.000 | 4.0 | 2 |
| arm_9cafad4e317c | /base/biblioness/bibs_location/country + /p/p/place_of_birth => /p/p/nationality | 1 | 0.000 | 0.000 | 0.0 | 0 |

#### nationality: 「地理的なrelation」が選ばれにくかった理由（考察）

直感的には nationality には地理（居住地・出生地・地名包含など）の文脈が効きそうだが、このrunでは UCB が主に award/film 系armを選好した。
これは「意味の妥当性」ではなく「proxy reward（witness/coverage等）を最大化する」最適化になっていることと、データ分布に起因する可能性が高い。

観測事実（このrunの `arm_history.pkl` 集計）:
- nationality の arm=10 のうち、地理系（例: `/location/location/contains` + `/people/person/places_lived...` / `/people/person/place_of_birth` 等）を含むarmは 4/10 存在するが、**選択回数は合計5回（25iter×k_sel=3 の計75選択のうち）**と少ない。
- 地理系armの `mean_reward` 平均は **約10.25**。それ以外（award/film等）のarmは **約20.73**で、UCBの活用（exploit）が非地理armに寄りやすい。
- 実際に上位armは以下で、`mean_witness_sum` や `total_added` が大きい（= proxy rewardが伸びやすい）:
  - `arm_8fc0f2710e9e`: award_nominee + nationality（mean_reward 40.9 / total_added 189）
  - `arm_ae6ee4488c19`: actor→film + film_country（mean_reward 29.1 / total_added 134）

なぜ地理armの proxy reward が伸びにくいか（仮説）:
1) **witnessが出にくい（疎/条件が厳しい）**
  - `contains`×`places_lived` のようなルールは、country（nationality側）と location（居住地/出生地）を接続する必要があり、実データ上「該当するwitnessが取れるターゲット」が限定されやすい。
2) **“それっぽい”が同義ではない（ノイズ混入）**
  - 「居住地」は国籍と相関はあるが同義ではないため、witnessが得られても“判別に効く”とは限らない。proxyがwitness量中心だと、このズレを罰せず、結果として別タイプのarm（award/film）に負けやすい。
3) **proxy rewardの設計が“意味整合”を直接見ていない**
  - 現状は witness/coverage 等の量的指標が中心で、predicateの意味整合（「nationalityに対して地理がより自然」）を reward に入れていない。
  - そのため、award/film のような高接続・高頻度関係は「witnessが稼げる」→「報酬が高い」→「UCBが選ぶ」という経路で優位になりやすい。
4) **初期armプールの偏り**
  - `arms/initial_arms.json` に地理armは含まれるが、構成として award/film 系の方が “高supportで回りやすいルール” になっており、探索初期に差が付くとUCBはそのまま活用へ寄りやすい。

含意（改善アイデア）:
- banditの reward / prior に「ターゲットrelationとの意味整合（predicate類似度）」を入れる（既に行った predicate偏り分析とも整合）。
- “高接続で稼げる関係”への過適応を抑えるため、predicateグループごとの探索枠（geo/award/film等）を設ける、または witness を「多様性/新規性」で重み付けする。

### place_of_birth

- arm定義: [experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/arms/initial_arms.json](../../experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/arms/initial_arms.json)
- 報酬ログ: [experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/arm_run/iter_25/arm_history.pkl](../../experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_place_of_birth_head_incident_v1_20260127a/arm_run/iter_25/arm_history.pkl)

| arm_id | rule (short) | n_eval | mean_reward | recent_mean | sum_reward | total_added |
|---|---|---:|---:|---:|---:|---:|
| arm_346364a79725 | /p/p/places_lived./people/place_lived/location => /p/p/place_of_birth | 23 | 9.826 | 8.667 | 226.0 | 36 |
| arm_854095aad5ef | /location/hud_county_place/place + /p/p/place_of_birth => /p/p/place_of_birth | 22 | 9.545 | 9.667 | 210.0 | 0 |
| arm_51b0adb47adb | /location/hud_county_place/place + /p/p/place_of_birth => /p/p/place_of_birth | 21 | 8.762 | 9.000 | 184.0 | 0 |
| arm_2a2fc1fc5a51 | /location/hud_county_place/place + /p/p/places_lived./people/place_lived/location => /p/p/place_of_birth | 3 | 6.667 | 6.667 | 20.0 | 6 |
| arm_5cd45ee5f1b9 | /music/artist/origin => /p/p/place_of_birth | 1 | 4.000 | 4.000 | 4.0 | 2 |
| arm_66f1646d6579 | /location/hud_county_place/place + /p/p/places_lived./people/place_lived/location => /p/p/place_of_birth | 1 | 3.000 | 3.000 | 3.0 | 1 |
| arm_f4e33d5d1887 | /location/hud_county_place/place + /music/artist/origin => /p/p/place_of_birth | 1 | 3.000 | 3.000 | 3.0 | 1 |
| arm_287433ce5b60 | /location/us_county/county_seat + /p/p/place_of_birth => /p/p/place_of_birth | 1 | 1.000 | 1.000 | 1.0 | 0 |
| arm_596bcf388b89 | /location/us_county/county_seat + /p/p/place_of_birth => /p/p/place_of_birth | 1 | 1.000 | 1.000 | 1.0 | 0 |
| arm_2f3acbc44596 | /music/artist/origin + /p/p/places_lived./people/place_lived/location => /p/p/place_of_birth | 1 | 0.000 | 0.000 | 0.0 | 0 |

### profession

- arm定義: [experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_profession_head_incident_v1_20260127b/arms/initial_arms.json](../../experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_profession_head_incident_v1_20260127b/arms/initial_arms.json)
- 報酬ログ: [experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_profession_head_incident_v1_20260127b/arm_run/iter_25/arm_history.pkl](../../experiments/20260126_rerun1/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_profession_head_incident_v1_20260127b/arm_run/iter_25/arm_history.pkl)

| arm_id | rule (short) | n_eval | mean_reward | recent_mean | sum_reward | total_added |
|---|---|---:|---:|---:|---:|---:|
| arm_68ee3dd1584b | /a/award_nominee/award_nominations./a/award_nomination/award_nominee + /p/p/profession => /p/p/profession | 23 | 15.739 | 10.000 | 362.0 | 118 |
| arm_15670562d47d | /a/award_winner/awards_won./a/award_honor/award_winner + /p/p/profession => /p/p/profession | 21 | 8.333 | 5.667 | 175.0 | 55 |
| arm_ead03006d03a | /a/award_winner/awards_won./a/award_honor/award_winner + /p/p/profession => /p/p/profession | 16 | 7.625 | 8.667 | 122.0 | 48 |
| arm_53cf490ecaf9 | /base/popstra/celebrity/friendship./base/popstra/friendship/participant + /p/p/profession => /p/p/profession | 4 | 4.750 | 3.667 | 19.0 | 8 |
| arm_4c65276cb9d8 | /base/popstra/celebrity/dated./base/popstra/dated/participant + /p/p/profession => /p/p/profession | 4 | 4.250 | 2.333 | 17.0 | 8 |
| arm_6017b14bdad0 | /base/popstra/celebrity/dated./base/popstra/dated/participant + /p/p/profession => /p/p/profession | 2 | 4.000 | 4.000 | 8.0 | 4 |
| arm_463faf9acd70 | /a/award_nominee/award_nominations./a/award_nomination/award_nominee + /p/p/profession => /p/p/profession | 2 | 3.000 | 3.000 | 6.0 | 3 |
| arm_ee40a8a14144 | /p/p/profession + /people/profession/specialization_of => /p/p/profession | 1 | 2.000 | 2.000 | 2.0 | 0 |
| arm_b1442ab397de | /base/popstra/celebrity/friendship./base/popstra/friendship/participant + /p/p/profession => /p/p/profession | 1 | 0.000 | 0.000 | 0.0 | 0 |
| arm_a4b9cc0f2756 | /influence/influence_node/influenced_by + /p/p/profession => /p/p/profession | 1 | 0.000 | 0.000 | 0.0 | 0 |

## 観察（重要）: 追加トリプルはターゲットそのものではない

3関係すべてで `direct_target_triples_added = 0`（ターゲット triple そのものは追加されていない）。
従って、ターゲットのスコア変化は「追加された周辺トリプルにより、埋め込み空間/ランキングが間接的に変形した結果」と解釈するのが妥当。

また head_coverage（ターゲットheadが追加トリプルheadでどれだけ被覆されたか）も、勝敗と単純には一致しない。
例: nationality は Random の head_coverage が高い（0.9）が UCB が勝ち、place_of_birth は UCB の head_coverage が高い（0.45）が Random が勝つ。

## なぜ UCB が target_score で常勝しないか（考察）

### 1) target_score は「モデルのスコア校正（calibration）変化」に弱い
summary.json の `target_score_change` は各モデルの raw score を合算した値で、学習後のスコア分布の平行移動/スケール変化（校正のズレ）を直接受ける。
そのため「ランキングや局所的な相対関係は改善しているが、raw合計は別条件が勝つ」ことが起き得る。

この現象は profession で顕著:
- raw target_score_change は Random が勝ち
- しかし minmax(train) 正規化では UCB が Δmean / improved_frac ともに明確に勝つ

→ 「raw target_score の勝敗」だけで判断すると誤解し得るため、minmax(train) 再スコアを併記するのが妥当。

### 2) 追加は“間接効果”なので、改善と悪化が混在する
ターゲット triple 自体は追加されていないため、追加トリプルはターゲット周辺の制約（近傍構造）を増やす。
このとき PairRE のような関係パラメータ化モデルでは、
- あるターゲット（h,r,t）を押し上げる追加が
- 別ターゲット（h',r,t'）や同一関係内の競合 tail の相対順位を押し下げる
というトレードオフが起きる。

### 3) “被覆（coverage）”は必要条件だが十分条件ではない
head_coverage が高くても、追加された文脈がターゲット関係の識別に寄与しない（あるいはノイズ/競合制約になる）場合、ターゲットは下がる。
特に place_of_birth は追加数が45と小さく、追加セットの性質の差が結果に直結しやすい。

### 4) UCB は proxy reward 最適化であり、target_score 最適化ではない
UCB は arm ごとの proxy 指標（evidence/witness/coverage 等）から探索・活用をするため、
- proxy が増える（新規証拠が取れる）
- しかし KGE目的（ランキング損失）上は一部ターゲットを悪化させる
というズレが残り得る。

### 5) 追加トリプルの「意味（predicateの種類）」がターゲット改善に効く
head_coverage は「ターゲットheadに文脈が追加されたか」を測るが、**どのpredicateの文脈か**までは区別しない。
rerun1 の追加トリプルを predicate 別に集計し、さらに「ターゲット改善head（mean Δ>0）に付いた追加」vs「悪化head（mean Δ≤0）に付いた追加」で分けると、いくつかの偏りが観測できる。

（詳細: [analysis_added_triples_predicates_20260128.md](../../experiments/20260126_rerun1/analysis_added_triples_predicates_20260128.md)）

観測された例（抜粋）:
- nationality / UCB:
  - 追加の大半が `/award/...` と `/film/actor/film...` に集中。
  - `/film/actor/film...` は悪化head側での比率が高い（share_improved 52.6% vs share_degraded 72.2%）。
  - 逆に `/award/.../nominated_for` は改善head側での比率が高い（32.6% vs 16.7%）。
  - 解釈: 「追加文脈が多い」だけでは不足で、nationality の識別に寄与しない（または競合する）文脈が混ざると一部headのスコアを押し下げ得る。

- place_of_birth / UCB:
  - 追加のほとんどが `/people/person/places_lived.../location`（42/45）。
  - 解釈: place_of_birth に対して places_lived は近いが同義ではなく、追加が単一タイプに偏ると「出生地 vs 居住地」の混同を招き、改善しないheadが残り得る。

- profession / UCB:
  - 追加の多くが `/award/.../award_nominee` と `/award/.../award_winner`。
  - 悪化headは少数（8）だが、その側では `/award/.../award_nominee` の比率が極端に高い（100%）。
  - 一方で `/award/.../award_winner` は改善head側に偏る（27.9% vs 0.0%）。
  - 解釈: award関連でも「どの役割のエッジか」で有益性が異なる可能性がある（少なくとも“常に良い”ではない）。

含意:
- 今回の UCB（proxy最適化）では、meaningful なトリプル追加（ターゲット関係に効く文脈の追加）を直接最適化できていない可能性がある。
- 次の改良候補として、bandit の reward / prior / 制約に predicate レベルの意味（ターゲット関係との整合）を組み込む余地がある。

## 上昇/下降ターゲット例（minmax(train) Δ）

ここでは「どのターゲットが上がり/下がったか」の代表例を抜粋し、追加された周辺トリプル（文脈）も添えて示す。

- 完全一覧（per-target Δ等）: [analysis_ucb_vs_random_seed0_3relations_20260128.json](../../experiments/20260126_rerun1/analysis_ucb_vs_random_seed0_3relations_20260128.json)
- 本セクションの例をそのままMarkdown化した出力: [analysis_combined_examples_with_context_20260128.md](../../experiments/20260126_rerun1/analysis_combined_examples_with_context_20260128.md)

選び方:
- UCB と Random(seed=0) の両条件から例を集約し、minmax(train) 正規化Δの大小で並べる（= 手法ラベルは付けない）。
- 「追加文脈」は、追加トリプルの subject がターゲットトリプル内の2つのエンティティのどちらかに一致するものをサンプル表示。
- （該当なし）は「どちらのエンティティをsubjectに持つ追加トリプルが見つからなかった」を意味する。

### nationality

#### 上昇例

- Irwin Winkler — /people/person/nationality — United States of America (Δ=+0.102)
  - 追加文脈（サンプル）:
    - Irwin Winkler — /award/award_nominee/award_nominations./award/award_nomination/nominated_for — Rocky V
    - Irwin Winkler — /award/award_nominee/award_nominations./award/award_nomination/nominated_for — Raging Bull
    - Irwin Winkler — /award/award_nominee/award_nominations./award/award_nomination/nominated_for — Rocky
- Tom Skerritt — /people/person/nationality — United States of America (Δ=+0.072)
  - 追加文脈（サンプル）:
    - Tom Skerritt — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Ray Walston
    - Tom Skerritt — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Don Cheadle
    - Tom Skerritt — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Fyvush Finkel
- Ian Holm — /people/person/nationality — England (Δ=+0.071)
  - 追加文脈（サンプル）:
    - Ian Holm — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Cate Blanchett
    - Ian Holm — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Miranda Otto
    - Ian Holm — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Danny Huston
- Josh Hartnett — /people/person/nationality — United States of America (Δ=+0.070)
  - 追加文脈（サンプル）:
    - Josh Hartnett — /award/award_nominee/award_nominations./award/award_nomination/nominated_for — Pearl Harbor
    - Josh Hartnett — /film/actor/film./film/performance/film — Sin City
    - Josh Hartnett — /film/actor/film./film/performance/film — The Black Dahlia

#### 下降例

- Raj Kapoor — /people/person/nationality — India (Δ=-0.060)
  - 追加文脈（サンプル）:
    - Raj Kapoor — /people/person/sibling_s./people/sibling_relationship/sibling — Shashi Kapoor
    - Raj Kapoor — /people/person/sibling_s./people/sibling_relationship/sibling — Shammi Kapoor
- Ron W. Miller — /people/person/nationality — United States of America (Δ=-0.052)
  - 追加文脈（サンプル）:
    - Ron W. Miller — /people/person/profession — Film Producer-GB
    - Ron W. Miller — /people/person/spouse_s./people/marriage/type_of_union — Marriage
- Naomi Watts — /people/person/nationality — Australia (Δ=-0.052)
  - 追加文脈（サンプル）:
    - Naomi Watts — /film/actor/film./film/performance/film — Ned Kelly
    - Naomi Watts — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Nicole Kidman
    - Naomi Watts — /people/person/profession — Film Producer-GB
- KT Tunstall — /people/person/nationality — Scotland (Δ=-0.050)
  - 追加文脈（サンプル）:
    - KT Tunstall — /people/person/place_of_birth — Edinburgh
    - KT Tunstall — /people/person/places_lived./people/place_lived/location — Edinburgh
    - KT Tunstall — /music/artist/track_contributions./music/track_contribution/role — Guitar

### place_of_birth

#### 上昇例

- Jon Lovitz — /people/person/place_of_birth — Los Angeles (Δ=+0.078)
  - 追加文脈（サンプル）: （該当なし）
- Jeffrey Combs — /people/person/place_of_birth — Oxnard (Δ=+0.076)
  - 追加文脈（サンプル）:
    - Jeffrey Combs — /people/person/places_lived./people/place_lived/location — Oxnard
- Cedric the Entertainer — /people/person/place_of_birth — Jefferson City (Δ=+0.073)
  - 追加文脈（サンプル）:
    - Cedric the Entertainer — /people/person/places_lived./people/place_lived/location — Jefferson City
- Jello Biafra — /people/person/place_of_birth — Boulder (Δ=+0.069)
  - 追加文脈（サンプル）:
    - Jello Biafra — /people/person/places_lived./people/place_lived/location — Boulder

#### 下降例

- Taraji P. Henson — /people/person/place_of_birth — Washington, D.C. (Δ=-0.067)
  - 追加文脈（サンプル）:
    - Taraji P. Henson — /people/person/places_lived./people/place_lived/location — Washington, D.C.
    - Taraji P. Henson — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Cate Blanchett
- Loudon Wainwright III — /people/person/place_of_birth — Chapel Hill (Δ=-0.061)
  - 追加文脈（サンプル）: （該当なし）
- Carmine Infantino — /people/person/place_of_birth — New York City (Δ=-0.054)
  - 追加文脈（サンプル）: （該当なし）
- Hogan Sheffer — /people/person/place_of_birth — York (Δ=-0.051)
  - 追加文脈（サンプル）:
    - Hogan Sheffer — /people/person/places_lived./people/place_lived/location — York

### profession

#### 上昇例

- Justin Bieber — /people/person/profession — Musician-GB (Δ=+0.109)
  - 追加文脈（サンプル）:
    - Justin Bieber — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Ludacris
    - Justin Bieber — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Demi Lovato
    - Justin Bieber — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Nick Jonas
- Justin Bieber — /people/person/profession — Singer-songwriter-GB (Δ=+0.109)
  - 追加文脈（サンプル）:
    - Justin Bieber — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Ludacris
    - Justin Bieber — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Demi Lovato
    - Justin Bieber — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Nick Jonas
- Justin Bieber — /people/person/profession — Actor-GB (Δ=+0.107)
  - 追加文脈（サンプル）:
    - Justin Bieber — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Ludacris
    - Justin Bieber — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Demi Lovato
    - Justin Bieber — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Nick Jonas
- Justin Bieber — /people/person/profession — Artist-GB (Δ=+0.106)
  - 追加文脈（サンプル）:
    - Justin Bieber — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Ludacris
    - Justin Bieber — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Demi Lovato
    - Justin Bieber — /base/popstra/celebrity/friendship./base/popstra/friendship/participant — Nick Jonas

#### 下降例

- Meshell Ndegeocello — /people/person/profession — Multi-instrumentalist-GB (Δ=-0.067)
  - 追加文脈（サンプル）:
    - Meshell Ndegeocello — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Chaka Khan
    - Meshell Ndegeocello — /film/actor/film./film/performance/film — Standing in the Shadows of Motown
    - Meshell Ndegeocello — /people/person/place_of_birth — Berlin
- Meshell Ndegeocello — /people/person/profession — Songwriter-GB (Δ=-0.061)
  - 追加文脈（サンプル）:
    - Meshell Ndegeocello — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Chaka Khan
    - Meshell Ndegeocello — /film/actor/film./film/performance/film — Standing in the Shadows of Motown
    - Meshell Ndegeocello — /people/person/place_of_birth — Berlin
- Meshell Ndegeocello — /people/person/profession — Bassist (Δ=-0.061)
  - 追加文脈（サンプル）:
    - Meshell Ndegeocello — /award/award_nominee/award_nominations./award/award_nomination/award_nominee — Chaka Khan
    - Meshell Ndegeocello — /film/actor/film./film/performance/film — Standing in the Shadows of Motown
    - Meshell Ndegeocello — /people/person/place_of_birth — Berlin
- Boris Leven — /people/person/profession — Production Designer (Δ=-0.060)
  - 追加文脈（サンプル）: （該当なし）

## 付録: 実験成果物リンク

- 旧（experiments配下に生成した統合サマリ）: [experiments/20260126_rerun1/summary_ucb_vs_random_seed0_3relations_20260128.md](../../experiments/20260126_rerun1/summary_ucb_vs_random_seed0_3relations_20260128.md)

