# REC-20260201-EXPERIMENT_RESULTS_SUMMARY-001: 実験結果サマリ（Random baseline / UCB vs Random（3関係）/ LLM-policy nationality の累積KGE再学習評価）

- 作成日: 2026-02-01
- 最終更新日: 2026-02-01

## 概要

本ドキュメントは、以下3つの実験記録を「目的 / 実験条件 / 結果 / 考察 / 結論」の形式で統合要約する。
表記（用語、評価指標の書き方など）は [docs/output/aaai_draft_20260124_ja.md](../output/aaai_draft_20260124_ja.md) に合わせる。

対象（元ファイル）:

- [REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001](REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001.md)
- [REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001](REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md)
- [REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001](REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001.md)

補足:
- 本プロジェクトでは、最終目的を KGE（知識グラフ埋め込み）に基づくリンク予測精度（Hits@k、MRRなど）の改善として扱う。
- 一方で、反復中に得られる「追加トリプル（evidence）」はターゲット関係そのものではなく、周辺事実（文脈）であるため、評価は間接効果（埋め込み空間・ランキングの変形）として解釈する必要がある。

記号整理（教授向けの数学的表記）:

- 知識グラフを $G=(\mathcal{E},\mathcal{R},\mathcal{T})$（エンティティ集合、関係集合、トリプル集合）とする。
- ターゲット関係を $r^*\in\mathcal{R}$ とし、評価対象トリプル集合を $\mathcal{T}^*:=\{(h,r^*,t)\in\mathcal{T}\}$ とする。
- ターゲット entity（典型的には head 側）集合を $S\subseteq\mathcal{E}$ とする。
- incident triples（周辺文脈）の集合を

$$
\mathrm{Inc}(S):=\{(h,r,t)\in\mathcal{T}\mid h\in S\ \lor\ t\in S\}
$$

と定義する。
- 文脈除去（欠損の導入）を、ある除去集合 $\mathcal{D}\subseteq\mathcal{T}$ を用いて $\mathcal{T}_0:=\mathcal{T}\setminus\mathcal{D}$ と書く。
- 本プロジェクトにおける「追加候補集合」は、概念的には $\mathcal{C}:=\mathcal{D}$（除去したトリプルの集合）として扱う。
- 反復で得られる追加集合を $A_i\subseteq\mathcal{C}$ とし、累積 union を $U_{1:k}:=\bigcup_{i=1}^{k} A_i$ とする。

データセット生成（共通の考え方）:

- 本サマリで扱うテストデータセットは、FB15k-237 をベースに、特定のターゲット関係（例: /people/person/nationality）に対して target entities を選び、ターゲット周辺の文脈（incident triples）を部分的に除去して「欠損・疎な状況」を擬似的に作る。
- 生成仕様（厳密な定義・出力物の意味）は、[docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md](../rules/RULE-20260119-MAKE_TEST_DATASET-001.md) に従う。
- 生成物のうち、(i) target_triples は評価対象（$\mathcal{T}^*$）、(ii) train は追加前KG（$\mathcal{T}_0$）、(iii) 除去トリプル集合は追加候補（$\mathcal{C}=\mathcal{D}$）として利用する。

---

## 1. Random baseline（UCB追加 vs 同数ランダム追加）

参照（元ファイル）: [REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001](REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001.md)

### 1.1 目的

KG-FIT（PairRE）において、UCB（arm-runで選ばれた追加トリプル）による改善が、同じ追加候補集合 $\mathcal{C}$ から同数をランダムに追加した場合より良いかを検証する。
主指標は target triples の再スコア（min-max 正規化）であり、ランキング指標（Hits@k、MRR）は副指標とする。

### 1.2 実験条件

- 埋め込み: KG-FIT（PairRE）
- データセット（nationality v3 / 部分文脈削除）:
  - FB15k-237（train）をベースに、ターゲット関係を /people/person/nationality とし、head 側から target entities（人物）を自動選択（100件）。
  - ターゲット関係以外の関係で到達できる近傍エンティティを集め、その一部をサンプリング（drop_ratio=0.7, remove_preference=both）。
  - サンプリングされた近傍エンティティに incident なトリプル（target relation を含む）を train から除去し、文脈を希薄化（include_target=true）。
  - ここで除去されたトリプル集合を $\mathcal{D}$ とし、追加候補集合 $\mathcal{C}:=\mathcal{D}$（ランダム追加の母集団）として用いる。
- ターゲット関係: /people/person/nationality
- 追加候補（ランダム追加の母集団）: $\mathcal{C}$
- 追加トリプル数: UCB参照runに合わせて n_added_used = 10232
- 比較条件:
  - 条件U（参照）: UCB（arm=10, priors=off）の最終追加集合
  - 条件R（新規）: 追加候補集合 $\mathcal{C}$ から非復元抽出（seed=0..3）で同数追加
- 評価:
  - summary 指標（target_score_change、Hits@k、MRRなど）
  - min-max 正規化（各afterモデルの train に対する 0..1 正規化）で target 再スコア差分（after-before）を算出

### 1.3 結果

min-max 正規化 target 再スコア（主指標）:

- usable targets（全モデルで known な共通集合）: 106 / 117
- UCB:
  - Δmean = +0.041186
  - improved_frac = 0.924528
- Random（seed0..3）:
  - Δmean は全seedで負（例: seed2 = -0.170786）
  - improved_frac は全seedで 0
  - Δmean aggregate: mean = -0.153068, std = 0.0160266

追加トリプル（predicate）傾向（UCBの最終追加集合、上位10）:

- 総追加数: 10232
- 上位 predicate（件数）: 
  - /award/award_nominee/award_nominations./award/award_nomination/award_nominee（1992）
  - /award/award_winner/awards_won./award/award_honor/award_winner（1218）
  - /film/actor/film./film/performance/film（731）
  - /award/award_nominee/award_nominations./award/award_nomination/nominated_for（417）
  - /film/film/release_date_s./film/film_regional_release_date/film_release_region（363）
  - /people/person/places_lived./people/place_lived/location（324）
  - /location/location/contains（315）

arm 選択傾向（25iter, k_sel=3, 計75選択）:

- 選択回数上位3 arm（選択回数）:
  - arm_b9672dabe85d（23）: award_nominee + nationality（bodyに /award/... と /people/person/nationality を含む）
  - arm_6af063aed93b（19）: actor→film + film_country（bodyに /film/... を含む）
  - arm_8b7b14313003（15）: award_winner + nationality（bodyに /award/... と /people/person/nationality を含む）

### 1.4 考察

- 追加候補集合 $\mathcal{C}$ からの無差別追加（ランダム）は、target に対して一貫して悪化方向（min-max Δmean < 0）であり、「候補集合の中には target 最適化に不利な成分が多い」可能性が高い。
- UCBの選別（arm-runで得た追加集合）は、少なくとも本条件では target 再スコア改善に強く寄与している。
- 一方で ranking 指標（Hits@k、MRR）は seed により挙動が混在し、target 指標と一致しないケースがあり、目的関数（target最適化）と一般的なリンク予測性能（MRR/Hits）の間のトレードオフが示唆される。

補足（本実験の傾向）:
- 追加集合は award / film 系 predicate が大きな割合を占めており、UCBの proxy reward が「高頻度・高接続な関係」を拾いやすい可能性がある。

### 1.5 結論

- 観測: seed0..3 の範囲では、UCB（arm=10/priors=off）の追加は、同数ランダム追加より min-max 正規化の target 再スコア改善が明確に大きい。
- 含意: 候補集合からの「選別」が重要であり、無差別追加は target 最適化に不利になり得る。
- 次アクション: Randomのseedを追加（seed4以降）して分散推定を強化し、ranking指標（MRR/Hits）とのトレードオフを系統的に評価する。

---

## 2. UCB vs Random（3関係、incident除去データ、rerun1）

参照（元ファイル）: [REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001](REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md)

### 2.1 目的

incident triples 除去データ（nationality / place_of_birth / profession）に対して、KG-FIT（PairRE）で
- UCB（arm=10, priors off）で追加→再学習→評価
- 同数の Random（seed=0）で追加→再学習→評価

を比較し、(a) `summary.json` 指標、(b) min-max 正規化の per-target 再スコアリング（Δ集計）で勝敗を整理する。
加えて、UCBが常に勝たない理由を、追加トリプル（predicate）の偏りや proxy reward とのズレの観点から考察する。

### 2.2 実験条件

- 埋め込み: KG-FIT（PairRE）
- データセット（head_incident_v1 / ターゲットentity incident除去）:
  - 各ターゲット関係（nationality / place_of_birth / profession）ごとに、FB15k-237（train）をベースに、head 側から target entities（人物）を自動選択（100件）。
  - 選ばれた target entities に incident なトリプルを train から除去して文脈を希薄化（removal_mode=target_incidents）。
  - 除去トリプル集合を $\mathcal{D}$ とし、追加候補集合 $\mathcal{C}:=\mathcal{D}$（ランダム追加の母集団）として用いる。
- 比較条件:
  - 条件U: UCB（arm=10, priors=off）
  - 条件R: 同数ランダム追加（seed=0）
- 追加トリプル数（n_added_used）:
  - nationality: 445
  - place_of_birth: 45
  - profession: 244
- 評価:
  - summary の変化量（target_score_change、Hits@10_change、MRR_change など）
  - min-max 正規化 per-target Δ の集計（Δmean、improved_frac）



### 2.3 結果

`summary.json`（after retrain/eval）:

| relation | n_added_used | target_score_change (UCB / Random / winner) | mrr_change (UCB / Random / winner) | hits@10_change (UCB / Random / winner) |
|---|---:|---|---|---|
| nationality | 445 | 18.4574 / 16.5572 / UCB | -0.00183956 / -0.00413892 / UCB | -0.0149415 / -0.000245749 / Random |
| place_of_birth | 45 | -1.50677 / 2.17452 / Random | -0.000631236 / 0.00318363 / Random | 0.00388016 / 0.00574656 / Random |
| profession | 244 | 2.47249 / 3.29134 / Random | -0.00127502 / -0.00194813 / UCB | 0.0127908 / 0.00838817 / UCB |

min-max 正規化ターゲット再スコア（per-target Δの集計）:

| relation | Δmean (UCB / Random / winner) | improved_frac (UCB / Random / winner) | n_targets |
|---|---|---|---:|
| nationality | 0.0132431 / 0.00370567 / UCB | 0.692308 / 0.555556 / UCB | 117 |
| place_of_birth | 0.0111781 / 0.0183667 / Random | 0.67 / 0.74 / Random | 100 |
| profession | 0.0396948 / 0.0143782 / UCB | 0.909091 / 0.710744 / UCB | 363 |

### 2.4 考察

- raw の `target_score_change` は「スコア校正（calibration）の平行移動/スケール変化」を強く受けるため、勝敗がランキング改善と一致しない場合がある。
  - 例: profession は raw target_score_change では Random が勝つが、min-max 正規化（Δmean / improved_frac）では UCB が明確に勝つ。
- 追加はターゲット関係そのものではなく周辺事実の追加であるため、改善と悪化が混在しやすい（同一関係内の競合 tail の相対順位変動など）。
- UCB は proxy reward（witness/coverage 等）を最適化しており、KGE目的（ランキング損失）や「意味整合（target relation に効く文脈）」を直接最適化していない。
  そのため、proxy が稼げる predicate（高頻度・高接続）へ偏ると、一部のターゲットでは逆効果になり得る。
- predicate 分析から、同じ「award/film 系」でも有益性が一様ではない（例: profession では `/award/.../award_winner` が改善側に偏る一方、`/award/.../award_nominee` は悪化側に偏る等）。

arm / predicate の傾向（元記録の要点のみ）:

- nationality（head_incident_v1）: award / film 系の arm が優勢になりやすく、地理系（places_lived / place_of_birth / contains）arm は選択回数が相対的に少ない。
- place_of_birth（head_incident_v1）: 追加トリプルが /people/person/places_lived./people/place_lived/location に強く偏りやすく、「出生地」と「居住地」の非同値性がノイズ要因になり得る。
- profession（head_incident_v1）: award 系（nominee / winner）の追加が中心だが、winner と nominee で改善寄与が異なる可能性がある。

### 2.5 結論

- 観測: 3関係比較では、UCBが常勝ではない（place_of_birth では Random が優位）。一方で、評価指標としては min-max 正規化の per-target 再スコアが raw target_score より解釈可能性が高い。
- 含意: 「proxy reward 最適化」と「KGE評価（Hits@k/MRR）・target再スコア」の間にズレがあり、predicate レベルの意味整合を選別へ組み込む余地がある。
- 次アクション: bandit の prior/reward へ predicate の意味整合（ターゲット関係との関連性）を導入し、place_of_birth などでの “それっぽいが同義でない” 文脈（places_lived偏重）の悪影響を抑制する。

---

## 3. LLM-policy nationality の累積追加トリプルで KGE 再学習（iter1..3 vs iter1..final、同数Random対照）

参照（元ファイル）: [REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001](REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001.md)

### 3.1 目的

LLM-policy arm-run（nationality/head_incident_v1）の出力から、
- iter1..3 累積unionの追加トリプル
- iter1..final 累積unionの追加トリプル

をそれぞれ KG に加えて KGE を再学習し、target triples のスコア変化とリンク予測指標（Hits@k、MRR）の変化を評価する。
また、同数トリプルをランダムに選んで加える対照実験を行い、「取得トリプルが単なるランダム追加より有効か」を検証する。

### 3.2 実験条件

- 埋め込み: KG-FIT（PairRE）
- データセット: 2章と同様の head_incident_v1（ターゲット関係= /people/person/nationality）。
- before モデル: 上記データセット（追加前KG）で学習した KG-FIT（PairRE）を固定。
- 再学習設定: num_epochs=100（beforeと揃える）
- リーク防止: 追加データからターゲットpredicate（/people/person/nationality）を除外
- 比較条件（累積union版）:
  - iter1..3 union: 159件（actual vs 同数random）
  - iter1..final union: 887件（actual vs 同数random）

### 3.3 結果

before（共通）:

- target score (before): -736.4064
- Hits@10 (before): 0.290229
- MRR (before): 0.128665

actual - random（差分で提示）:

- iter1..3 union（159件）:
  - target score Δ=+19.8420
  - Hits@10 Δ=+0.013713
  - MRR Δ=+0.004195
- iter1..final union（887件）:
  - target score Δ=-14.9608
  - Hits@10 Δ=+0.000147
  - MRR Δ=-0.002824

### 3.4 考察

- 少量〜中量（iter1..3）の追加では、actual が random よりも改善しており、「有益な証拠追加」が成立し得る。
- 一方で大量（iter1..final）では、actual が random に対して target score と MRR で劣後しており、追加集合が大きくなるにつれてノイズ混入（意味不整合な文脈）が支配的になる可能性がある。
- 実験途中で incremental（単発iteration）と cumulative（累積union）の定義不整合が発生しており、比較可能性の観点から「評価定義（incremental/cumulative）」を明示する標準化が重要である。

追加トリプル（predicate）傾向（LLM-policy, actual）:

- iter1..3 union（159件）上位 predicate（件数）:
  - /award/award_nominee/award_nominations./award/award_nomination/award_nominee（61）
  - /people/person/places_lived./people/place_lived/location（59）
  - /people/person/place_of_birth（39）
- iter1..final union（887件）上位 predicate（件数）:
  - /award/award_nominee/award_nominations./award/award_nomination/award_nominee（372）
  - /award/award_winner/awards_won./award/award_honor/award_winner（192）
  - /film/actor/film./film/performance/film（135）
  - /award/award_nominee/award_nominations./award/award_nomination/nominated_for（75）
  - /people/person/places_lived./people/place_lived/location（65）
  - /people/person/place_of_birth（48）

arm 選択傾向（25iter, k_sel=3, 計75選択）:

- 選択回数上位3 arm（選択回数）:
  - arm_ae6ee4488c19（13）: actor→film + film_country（bodyに /film/... を含む）
  - arm_8fc0f2710e9e（12）: award_nominee + nationality（bodyに /award/... と /people/person/nationality を含む）
  - arm_268fbd2e5a65（11）: contains + places_lived（地理系のbodyを含む）

### 3.5 結論

- 観測: nationality では、iter1..3（159件）の累積追加は同数randomより有効だったが、iter1..final（887件）の累積追加は同数randomより劣後した。
- 含意: 「追加すればするほど良い」ではなく、追加トリプル集合の品質（predicate構成など）を制御する必要がある。
- 次アクション: final union（887件）を predicate サブセット（例: 地理系/非地理系）に分割し、劣化寄与成分を切り分ける。random対照は seed を増やして分散を見積もる。

---

## 4. 総合まとめ（横断的な含意）

- (1) 意味的に有益な文脈追加の価値: 少量〜中量の「選別された追加」（例: UCBの選別、LLM-policyのiter1..3）は、同数ランダム追加より改善し得る。これは $\mathcal{C}$ からの追加が、常にノイズではなく「ターゲット関係の識別に有利な制約」を与え得ることを示唆する。
- (2) incident追加の限界: incident triples は $\mathrm{Inc}(S)$ として定義できるが、$\mathrm{Inc}(S)$ の中には“それっぽいが同義でない”文脈が混在する（例: place_of_birth に対する places_lived）。従って、incident を戻すだけでは「意味整合な証拠」が自動的に得られるとは限らない。
- (3) LLM-policy の示唆: LLM-policy は、初期の少量追加（iter1..3）で random より良い結果が得られ、かつ選択 arm に地理系（contains + places_lived）が上位に現れるなど、proxy依存とは異なる選別が入り得る。
- (4) 大量追加での悪化（ノイズ混入）: LLM-policy の累積 union を大きくすると（iter1..final）、random より target score / MRR が劣後する例が観測され、追加集合のサイズ増加に伴う品質劣化（predicate構成の偏り、競合制約の増加）が主要なリスクとなる。

次の焦点:
- $\mathcal{C}$ の中で、ターゲット関係 $r^*$ に対する「意味整合」を predicate レベル（あるいは rule レベル）で定量化し、reward/prior/制約に組み込む。
