---
marp: true
theme: default
paginate: true
size: 16:9
---

# LLM-policy（nationality）
## 累積追加トリプルでKGE再学習

- 日付: 2026-02-01
- 対象: FB15k-237 / nationality

参照レポート:
- docs/records/REC-20260201-LLM_POLICY_NATIONALITY_CUMULATIVE_KGE_RETRAIN-001.md

---

## 目的

- LLM-policy が選別した追加トリプル集合が、
  同数のランダム追加より KGE（KG-FIT + PairRE）を改善するかを検証
- 評価は **累積union** の追加集合で実施
  - 早期: iter1..3
  - 最終: iter1..final

---

## 実験方法（概観）

- ベースKG: FB15k-237 から target entities（人物）を選ぶ
- target entities に接続する周辺トリプルを訓練用KGから除去し、欠損状況を作る
- 除去したトリプル集合を「追加候補集合」$\mathcal{C}$ とみなす
- 追加後KGでKGEを再学習し、ターゲット集合を評価

詳細仕様:
- docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md
- docs/rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md

---

## 追加集合（actual / random）

- actual:
  - LLM-policy arm-run で受理された追加トリプル
  - 累積union $U_{1:k}$ = iter1..k の受理トリプル和集合（重複除去）

- random:
  - 追加候補集合 $\mathcal{C}$ から actual と同数をランダム抽出して追加

リーク防止:
- 追加データからターゲットpredicate `/people/person/nationality` を除外

---

## KGEと評価

- 埋め込み: KG-FIT（PairRE）
- エポック数: before/after ともに 100
- 指標:
  - target score（ターゲット集合の集約スコア）
  - Hits@10
  - MRR

提示形式:
- **actual − random（差分）** を主要結果として提示

---

## 主要結果（before）

before（共通）

- target score: -736.4064
- Hits@10: 0.290229
- MRR: 0.128665

---

## 主要結果（actual − random）

| 条件 | 追加件数 | target score Δ | Hits@10 Δ | MRR Δ |
|---|---:|---:|---:|---:|
| iter1..3 union | 159 | +19.8420 | +0.013713 | +0.004195 |
| iter1..final union | 887 | -14.9608 | +0.000147 | -0.002824 |

---

## 考察（結果に基づく）

- 少量〜中量（iter1..3, 159件）:
  - actual が同数randomより **一貫して改善**（3指標すべてで正）

- 大量（iter1..final, 887件）:
  - actual が同数randomに対して **target score と MRR で劣後**

- Hits@10 は最終条件で差がほぼ0
  - 差が小さい領域では random の分散推定（seed増やす）が重要

---

## 追加集合の構成（俯瞰）

- 追加トリプルの上位 predicate は、主に以下に集中:
  - award 系
  - film 系
  - places_lived / place_of_birth 系

含意:
- 追加集合のサイズ増加に伴い品質が維持されない可能性
- サイズ上限や predicate/ルール単位の品質制御が必要

---

## 内訳: iter1..3 vs iter4..25

目的:
- 早期（iter1..3）と後期（iter4..25）で
  - どの arm が選ばれ
  - どのトリプルが取得されたか
  を比較し、最終条件の劣化要因を説明する

---

## arm選択（回数）

前提:
- k_sel = 3
- iter1..3: 9選択
- iter4..25: 66選択

観測:
- iter1..3: 地理系armが 8/9
- iter4..25: 地理系armが 17/66 まで低下

---

## iter1..3 の上位arm（例）

地理系が中心

- arm_268fbd2e5a65（3回）
  - /location/location/contains + /people/person/places_lived.../location
- arm_6a06bdebf12f（2回）
  - /base/biblioness/bibs_location/country + /people/person/places_lived.../location
- arm_711c97af9f2c（2回）
  - /location/location/contains + /people/person/place_of_birth

---

## iter4..25 の上位arm（例）

award/film 系が中心

- arm_ae6ee4488c19（13回）
  - /film/actor/film... + /film/film/country
- arm_8fc0f2710e9e（11回）
  - /award/award_nominee.../award_nominee + /people/person/nationality
- arm_9aee1a60dfa2（9回）
  - /award/award_nominee.../award_nominee + /people/person/nationality

---

## 取得トリプル（predicate分布）: iter1..3

iter1..3（159件）

- /award/award_nominee.../award_nominee: 61
- /people/person/places_lived.../location: 59
- /people/person/place_of_birth: 39

地理系（places_lived, place_of_birth, /location/*）:
- 98 / 159

---

## 取得トリプル（predicate分布）: iter4..25

iter4..25（728件）

- /award/award_nominee.../award_nominee: 311
- /award/award_winner.../award_winner: 192
- /film/actor/film.../film: 135
- /award/award_nominee.../nominated_for: 75

地理系:
- 15 / 728（place_of_birth 9, places_lived 6）

---

## （理由）地理系候補の枯渇

候補集合 $\mathcal{C}$ のうち地理系候補 $\mathcal{C}_{geo}$ を定義し集計

- $|\mathcal{C}| = 3496$
- $|\mathcal{C}_{geo}| = 140$（約4%）

消費:
- iter1..3 で 98件（約70%）を取得
- iter1..25 全体で 113/140（約81%）が既に取得

---

## 含意（枯渇が後半シフトを説明）

後半（iter4..25）で地理系取得が急減した主要因は

- 「LLM-policy が地理系を避けた」よりも
- **そもそも地理系候補が少なく、早期で大半を消費し、残余が枯渇した**

と解釈するのが筋が良い

---

## 結論

- 選別の価値:
  - iter1..3（159件）では actual が同数randomより改善（3指標すべてで正）

- “多ければ良い”の否定:
  - iter1..final（887件）では actual が random に対して target score / MRR で劣後

- 枯渇仮説（説明）:
  - 地理系候補は小さく、早期で大半を消費しやすい

---

## 運用提案

- 追加集合サイズに上限を設ける
- predicate/ルール単位で品質フィルタを導入する
- random 対照は複数seedで分散推定する

---

## 引用（関連実験と示唆）

- place_of_birth / profession でも UCB vs Random の比較が報告され、
  UCB が常に良いとは限らない
- proxy報酬に依存しやすい UCB を、意味的整合を加味し得る LLM-policy に変えると改善する可能性

参照:
- docs/records/REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md

---

# Appendix

## 参照ドキュメント

- docs/records/REC-20260201-LLM_POLICY_NATIONALITY_CUMULATIVE_KGE_RETRAIN-001.md
- docs/records/REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001.md
- docs/rules/RULE-20260119-MAKE_TEST_DATASET-001.md
- docs/rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md
