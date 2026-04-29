# REC-20260201-LLM_POLICY_NATIONALITY_CUMULATIVE_KGE_RETRAIN-001: LLM-policy（nationality）の累積追加トリプルでKGEを再学習し、同数Randomと比較する（主要結果レポート）

- 作成日: 2026-02-01
- 最終更新日: 2026-02-01

参照（元記録）:
- [REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001](REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001.md)

参照（仕様）:
- テストデータ生成: [docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md](../rules/RULE-20260119-MAKE_TEST_DATASET-001.md)
- 再学習評価プロトコル: [docs/rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md](../rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md)

## 1. 目的

LLM-policy により選別された追加トリプル集合が、同数のランダム追加よりも KGE（KG-FIT + PairRE）を改善するかを、
**累積union**の評価設定で検証する。

具体的には、LLM-policy arm-run の追加集合のうち
- 早期（iter1..3）の累積union
- 最終（iter1..final）の累積union

をそれぞれ追加して KGE を再学習し、同数ランダム追加を対照として、target triples のスコアとリンク予測指標（Hits@10, MRR）の差を評価する。

## 2. 実験方法（条件）

### 2.1 テストデータの作り方（概要）

FB15k-237 をベースに、ターゲット関係 $r^*$ を `/people/person/nationality` として次を行う:

- head 側から target entities（人物）を自動選択（100件）。
- 選ばれた target entities に直接接続する周辺トリプルを訓練用KGから除去し、欠損状況を作る。
- 除去したトリプル集合を追加候補集合 $\mathcal{C}$ とみなし、arm-run / random のいずれも $\mathcal{C}$ から追加を行う。
- 評価対象（target triples）は別途保持し、再学習したKGEでスコアとランキング指標を評価する。

厳密な定義・再現方法は [docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md](../rules/RULE-20260119-MAKE_TEST_DATASET-001.md) に従う。

### 2.2 追加データ（actual / random）の定義

- actual:
  - LLM-policy arm-run の各iterationで受理された追加トリプルを用いる。
  - 累積union $U_{1:k}$ は、iter1..k の受理トリプルの和集合（重複除去）とする。
- random:
  - 追加候補集合 $\mathcal{C}$ から、actual と同数のトリプルをランダムに選び追加する。
  - 可能な限り、重複トリプルや「既に訓練KGに存在するトリプル」を避ける（詳細はプロトコル文書参照）。

### 2.3 KGE と評価

- 埋め込みモデル: KG-FIT（PairRE）
- 学習（before）: 上記の欠損付きKGで学習したモデルを固定
- 学習（after）: before の訓練KGに追加集合（actual / random）を加えて再学習
- エポック数: 100（before/after で統一）
- リーク防止: 追加データからターゲットpredicate `/people/person/nationality` を除外
- 評価指標:
  - target score（target triples のスコア合計/平均に相当する集約値）
  - Hits@10
  - MRR

本レポートでは、**actual と random の差（actual - random）**を主要な比較として用いる（プロトコルは [docs/rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md](../rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md)）。

## 3. 結果と考察

### 3.1 結果（before と actual-random 差分）

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

### 3.2 考察（結果に基づく解釈）

- 少量〜中量（iter1..3, 159件）では、actual が同数randomより **一貫して良い方向**（target score / Hits@10 / MRR のいずれも actual-random > 0）であり、LLM-policy による追加集合が「ランダム追加よりも有益な情報を含む」可能性が高い。
- 一方で、追加集合を大きくすると（iter1..final, 887件）、actual が同数randomに対して **target score と MRR で劣後**し、量を増やすだけでは改善しない（むしろ悪化し得る）ことが示された。
- Hits@10 は iter1..final でも差分がほぼ0に近く、差が小さい領域では「評価の分散（randomのばらつき）」や「指標ごとの感度差」の影響が顕在化し得る。したがって、final条件は random seed を増やした分散推定が望ましい。

補助観察（追加集合の構成）:
- iter1..3 / iter1..final ともに、追加トリプルの上位 predicate は award / film / places_lived / place_of_birth 系が多い。
- 追加集合の拡大に伴い、（少なくとも一部の指標で）品質が維持されないため、追加集合のサイズ制御や、predicate/ルール単位での品質フィルタが必要である。

### 3.3 iter1..3 と iter4..25 の「arm選択」と「取得トリプル」

LLM-policy run を
- 早期: iter1..3
- 後期: iter4..25

に分けて、arm選択と取得トリプル（accepted）の傾向を整理する。

#### arm選択

（k_sel=3のため、iter1..3は計9選択、iter4..25は計66選択）

- iter1..3:
  - 地理系arm（例: `/location/*`, `places_lived`, `place_of_birth` を body に含む）の選択が 8/9 と支配的。
  - 選択回数上位arm（例）:
    - arm_268fbd2e5a65（3回）: /location/location/contains + /people/person/places_lived./people/place_lived/location
    - arm_6a06bdebf12f（2回）: /base/biblioness/bibs_location/country + /people/person/places_lived./people/place_lived/location
    - arm_711c97af9f2c（2回）: /location/location/contains + /people/person/place_of_birth

- iter4..25:
  - 地理系armの選択は 17/66 まで低下。
  - 選択回数上位arm（例）:
    - arm_ae6ee4488c19（13回）: /film/actor/film./film/performance/film + /film/film/country
    - arm_8fc0f2710e9e（11回）: /award/award_nominee/.../award_nomination/award_nominee + /people/person/nationality
    - arm_9aee1a60dfa2（9回）: /award/award_nominee/.../award_nomination/award_nominee + /people/person/nationality
    - arm_268fbd2e5a65（8回）: /location/location/contains + /people/person/places_lived./people/place_lived/location（地理系）

#### 取得トリプル（predicate分布）

- iter1..3（159件）:
  - 上位 predicate は3種類に集中:
    - /award/award_nominee/.../award_nomination/award_nominee（61）
    - /people/person/places_lived./people/place_lived/location（59）
    - /people/person/place_of_birth（39）
  - 地理系（`places_lived`, `place_of_birth`, `/location/*`）の取得は 98/159。

- iter4..25（728件）:
  - award/film 系が支配的:
    - /award/award_nominee/.../award_nomination/award_nominee（311）
    - /award/award_winner/.../award_honor/award_winner（192）
    - /film/actor/film./film/performance/film（135）
    - /award/award_nominee/.../award_nomination/nominated_for（75）
  - 地理系の取得は 15/728 と急減（内訳: place_of_birth 9, places_lived 6）。

#### （理由）筋の良い地理系候補の枯渇が主要因である可能性

追加候補集合を $\mathcal{C}$ とし、そのうち地理系候補部分集合を $\mathcal{C}_{\mathrm{geo}}$
（例: predicate が `/location/*` または `places_lived` / `place_of_birth`）とすると、観測は以下の通り:

- 候補集合の規模: $|\mathcal{C}|=3496$
- 地理系候補の規模: $|\mathcal{C}_{\mathrm{geo}}|=140$（候補全体の約4%）
- 地理系候補の消費:
  - iter1..3 で 98件を取得（$|\mathcal{C}_{\mathrm{geo}}|$ の約70%）
  - iter4..25 では 15件しか追加で取得されない
  - iter1..25 全体では地理系候補の約81%（113/140）が既に取得されている

このことから、後半（iter4..25）で地理系トリプルの取得が急減した主要因は、
LLM-policyの嗜好変化というよりも、**そもそも $\mathcal{C}_{\mathrm{geo}}$ が小さく、早期で大半が消費され、残余が枯渇した**ことにある可能性が高い。

## 4. 結論

本実験結果（actual-random差分）に基づく総括は以下である:

- 選別の価値: 早期の累積追加（iter1..3, 159件）では、LLM-policy の追加集合が同数randomより target score / Hits@10 / MRR の全てで改善した。したがって、追加候補集合からの「選別」には実効性がある。
- “多ければ良い”の否定: 追加を累積して増やすと（iter1..final, 887件）、同数randomに対して target score と MRR が悪化した。追加集合の品質管理なしに量を増やすのは危険である。
- 枯渇仮説（結果に基づく説明）: iter1..3 では地理系の取得が多い一方、iter4..25 では地理系の取得が急減した。候補集合内の地理系候補自体が少なく、早期で大半が消費されるため、後半で「筋の良い地理系候補が枯渇」しやすい。
- 運用含意: LLM-policy を主要戦略として用いる場合でも、(i) 追加集合のサイズ上限、(ii) predicate/ルール単位の品質フィルタ、(iii) random対照の複数seed化（分散推定）をプロトコルとして組み込むべきである。

## 5. 引用（関連実験と示唆）

> nationality 以外（place_of_birth / profession）でも、UCB と Random を比較する実験が報告されており、UCBが常に良い結果になるわけではない（関係によっては Random が優位）。
> したがって、proxy報酬に依存しやすい UCB を、意味的整合性を加味し得る LLM-policy に置き換えることで、改善する可能性がある。

参照:
- [REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001](REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md)
