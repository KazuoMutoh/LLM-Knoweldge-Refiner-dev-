# RULE-20260118-RELATION_PRIORS-001: relation priors（X_r）によるKGEフレンドな witness 重み付け

作成日: 2026-01-18
最終更新日: 2026-01-18

---

## 0. 目的とスコープ

本ドキュメントは、arm-based KG refinement における proxy 指標のうち **witness（支持構造）**を、
relation-level の事前スコア（relation priors; $X_r$）で重み付けするための設計ルール/標準を定義する。

狙い:
- witness が **ハブ関係**や **KGEで扱いにくい関係**によって水増しされる現象を抑制する
- 反復内ではKGEの再学習や再スコアリングを行わずに、witness の質（=KGE-friendly）を押し上げる
- 実験で再現可能な「外部から与えるprior（JSON）」として扱い、運用と検証を容易にする

スコープ:
- $X_r$ の構成要素（$X_2,X_3,X_4,X_7$）と、統合スコア $X$ の扱い
- $X_r$ を **witness計算にのみ適用**するルール（raw witness は診断用に保持）
- 事前計算（オフライン）と、統合ランナー（run_full_arm_pipeline）での計算・利用順序

非スコープ:
- $X_2..X_7$ の理論的背景の詳細（検討メモ参照）
- KGEモデルの内部設定や学習手順の詳細

---

## 1. 定義

### 1.1 relation prior $X_r$
各述語（predicate）$r$ に対し、$X_r \in [0,1]$ を定義する。

- $X_r$ が大きいほど「その関係が witness の根拠として **KGEフレンド**（幾何的に一貫・過度にハブらない等）」であるとみなす
- $X_r$ は **反復ループ内で更新しない**（before model と初期データからのオフライン算出）

### 1.2 統合スコア $X$
実装では、次の4指標の線形結合で統合スコア $X$ を作り、payload として保存する。

$$
X = \mathrm{clip}_{[0,1]}\Big( w_2 X_2 + w_3 X_3 + w_4 X_4 + w_7 X_7 \Big)
$$

- デフォルト重み: $w_7=1.0, w_2=w_3=w_4=0.0$（まず幾何的一貫性に寄せる）
- 重み総和が 0 の場合はフォールバック（実装では safe な既定値へ）

---

## 2. 指標の概要（実装に合わせた要点）

実装は [simple_active_refine/relation_priors_compute.py](../../simple_active_refine/relation_priors_compute.py) に準拠する。

- $X_2$（hubness抑制）: 人気ノード経由で水増しされやすい関係を低くする
- $X_3$（role coherence）: subject/object 側の役割が混線しにくい関係を高くする
- $X_4$（concentration）: 一部ノードに極端に集中する関係を低くする
- $X_7$（geometric consistency）: before KGE の埋め込みにおける幾何的整合性が高い関係を高くする

備考:
- $X_7$ は before model の entity embeddings を読むため、**before model のディレクトリ**が必要
- 指標は大規模データでも走るよう、relationごとにサンプリング上限（例: max_samples_x7_per_relation）を設ける

---

## 3. ファイル形式（relation_priors.json）

### 3.1 入力として受理する形式
arm-run 側は **predicate -> prior（float）** の辞書として利用する。
ただし運用上、次の2形式を受け入れる。

1) シンプル形式（直接）

- `{"/people/person/nationality": 0.93, ...}`

2) payload ラップ形式（計算スクリプト出力）

- `{"meta": {...}, "payload": {"/people/person/nationality": {"X": 0.93, "X2":..., ...}, ...}}`

ロードは [simple_active_refine/relation_priors.py](../../simple_active_refine/relation_priors.py) の `load_relation_priors()` に準拠し、
上記のような meta/payload を unwrap して `predicate -> X` に正規化する。

---

## 4. witness への適用ルール（重要）

### 4.1 raw witness と weighted witness

- raw witness: `count_witnesses_for_head(...)` の **置換数**（整数）
- weighted witness: raw witness を rule 単位の重みでスケールした **スコア**（実数）

実装（acquirer）は、各 rule $h$ に対して

$$
W(h) = \prod_{p \in \mathrm{body\_predicates}(h)} X_p
$$

を計算し、ターゲット $t$ に対する witness 数 $c_h(t)$ を

$$
\mathrm{weighted\_witness}(t) = \sum_{h \in a} W(h)\,c_h(t)
$$

として集計する。

- $X_p$ が未定義の場合は `default_relation_prior`（既定 1.0）を用いる
- priors を与えない場合は $W(h)=1$ として従来の witness を維持する

参照実装:
- witness取得側: [simple_active_refine/arm_triple_acquirer_impl.py](../../simple_active_refine/arm_triple_acquirer_impl.py)
- 報酬計算側: [simple_active_refine/arm_triple_evaluator_impl.py](../../simple_active_refine/arm_triple_evaluator_impl.py)

### 4.2 報酬への反映
報酬 $R(a)$ は「witness と evidence 数」の線形結合であり、witness には weighted witness を使う。

- `reward = witness_weight * witness_sum + evidence_weight * accepted_count`
- `diagnostics` として raw/weighted の両方を保存する（解析・デバッグ用）

狙い:
- prior を効かせるのは「代理報酬の一部（witness）」に限定し、
  raw witness（置換数）自体は観測値として保持して後で説明・分析可能にする

---

## 5. 事前計算と利用順序（運用ルール）

### 5.1 統合ランナーでの優先順位
[run_full_arm_pipeline.py](../../run_full_arm_pipeline.py) では次の優先順位で prior を解決する。

1. `--relation_priors_path`（明示指定）
2. `--compute_relation_priors` を指定した場合は、run_dir 配下へ計算してそれを使用
3. 上記が無い場合は、arm-run 内で `<dataset_dir>/relation_priors.json` があれば自動検出して使用

### 5.2 反復内での不変条件
- $X_r$ は反復内で更新しない（before model と初期データの関数）
- 反復ログの改善に応じて prior を更新したくなった場合は **別実験**として扱い、run_dir を分ける

---

## 6. 関連ドキュメント

- 検討メモ（背景・指標案）: [docs/external/KGEフレンドさを考慮したwitness評価の改善.md](../external/KGEフレンドさを考慮したwitness評価の改善.md)
- 設計記録（実装計画/実験への導入）: [docs/records/REC-20260117-WITNESS_EVAL-001.md](../records/REC-20260117-WITNESS_EVAL-001.md)
- armパイプライン仕様（どこで読み込まれ、どこへ保存されるか）:
  - [docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md](RULE-20260117-ARM_PIPELINE_IMPL-001.md)
  - [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
