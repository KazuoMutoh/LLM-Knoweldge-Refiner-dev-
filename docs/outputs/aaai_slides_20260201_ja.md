---
marp: true
theme: default
paginate: true
math: katex
size: 16:9
style: |
  section { font-size: 26px; }
  h1 { font-size: 46px; }
  h2 { font-size: 36px; }
  h3 { font-size: 30px; }
  .small { font-size: 22px; }
  .xsmall { font-size: 19px; }
  .tiny { font-size: 17px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .cols3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }
  .box { border: 1px solid #bbb; border-radius: 10px; padding: 12px 14px; }
  .muted { color: #555; }
  code { font-size: 0.92em; }
---

<!-- _class: lead -->
# 外部根拠に基づく
# ターゲット関係特化KGリファイン

## Hornルール誘導・Web候補拡張・代理報酬による逐次探索

<div class="muted">AAAI Draft Slides（日本語） / 2026-02-01</div>

---

## 本スライドのゴール

- 「ターゲット関係 $r^*$ のリンク予測」を改善するために、KG外から **根拠（evidence）** を集めて **観測そのもの** を増やす
- 何を取りに行くか（ルール群=arm）を **逐次意思決定（多腕バンディット）** として最適化
- 反復中は再学習が重いので、**代理報酬**で探索を回し、最後にまとめて再学習して評価

---

## 1. 背景：KG品質問題と外部根拠

- KGは欠損・誤り・陳腐化・矛盾を内在しやすい
- KG内部の構造だけでは埋まらない欠損がある（根拠がKG外に存在）
- KGが疎だと、リンク予測モデルが参照できる証拠が不足
- 外部情報源から根拠（evidence）を収集し、トリプルとして統合することは、観測を増やす直接策
- 外部は広大なので「探索→追加→方針更新」を反復する設計が自然

---

## 1.1 本研究の立場と貢献（要約）

- ターゲット関係に焦点を当て、外部情報源から evidence を取得してKGへ統合する反復枠組みを定式化
- Hornルールに基づく evidence 探索（acquire）＋ arm選択を逐次意思決定として扱い、予算下で適応更新
- Web由来候補統合：安定な識別子付与＋既存KGへの entity linking（類似検索＋LLM同一性判定）
- 反復後にKGE/TAKGEを再学習し、追加観測の効果を評価（代理報酬設計の実装指針）

---

## 2. 関連研究の整理（俯瞰）

<div class="cols">
<div class="box">

### (1) KGE/KGC
- TransE / DistMult / ComplEx / RotatE …
- 疎KG・KG外根拠に対する限界

### TAKG/TAKGE
- テキスト属性付きKG（TAKG）
- テキスト統合埋め込み（TAKGE）
- KG-FIT：テキスト＋階層正則化

</div>
<div class="box">

### (2) 外部知識取得・統合
- Open-world統合、信頼性推定
- RAG系：取得＋生成（主に下流）

### (3) HITL/アクティブ
- 予算下の検証・追加
- 監査可能性（provenance・ログ）

### (4) ルールベース
- AMIE系：規則抽出・説明
- 本研究：head生成ではなく、body側evidence探索の誘導に利用

</div>
</div>

---

## 本研究の位置づけ

- 目標：KGEを高度化するより、**外部根拠の統合で学習条件（観測の疎性）を直接改善**
- ルールは「headを確定追加」するためでなく、**body側の具体トリプル（evidence）を探索・収集**するための探索バイアス
- Web統合は provenance と名寄せを重視（監査可能性・運用性）

---

## 3. 問題設定（逐次意思決定）

- エンティティ集合 $\mathcal{E}$、関係集合 $\mathcal{R}$、トリプル $(h,r,t)$
- 初期トリプル集合（初期KG）$\mathcal{T}_0$
- ターゲット関係 $r^*\in\mathcal{R}$（例：国籍）
- 学習手続き $\theta_t=\mathrm{Train}(\mathcal{T}_t)$
- 評価関数 $\mathcal{M}(\theta,\mathcal{D}_{\mathrm{target}})$（Hits@k, MRR）

---

## 3.2 逐次追加の定式化

- 状態 $s_t=(\mathcal{T}_t,\theta_t)$
- 行動 $a_t\sim\pi(a\mid s_t)$（外部情報獲得の選択）
- 追加トリプル集合 $\Delta_t$ を得て $\mathcal{T}_{t+1}=\mathcal{T}_t\cup\Delta_t$
- 理想的報酬（評価増分）：

$$
 r_t=\mathcal{M}(\theta_{t+1},\mathcal{D}_{\mathrm{target}})-\mathcal{M}(\theta_t,\mathcal{D}_{\mathrm{target}})
$$

- ただしTAKGE再学習が高コスト → 反復中は **代理報酬**で評価・選択し、再学習は反復後/バッチ

---

## 3.3 目的と制約（予算）

$$
\max_{\pi}\;\mathbb{E}[\mathcal{M}(\theta_T,\mathcal{D}_{\mathrm{target}})]
$$

$$
\text{s.t. }\sum_{t=0}^{T-1} c(a_t,\Delta_t)\le B
$$

- $c$ は問い合わせ回数や追加数などで表現可能

---

<!-- _class: small -->
## 4. 提案手法：全体像

- 方針：ターゲット $(x,r^*,y)$ を直接確定追加しない
- 代わりに、$r^*$ を支持し得る **周辺事実（evidence）** を集めて観測を増やし、最後に再学習

> Running example（国籍予測）
> - $r^*=\texttt{/people/person/nationality}$
> - ターゲット $(p_0,r^*,c_0)$ は評価用（学習用KGから除去）
> - 反復中は head $(p_0,r^*,c_0)$ を追加しない
> - 出生地/居住地/所在地などの evidence を追加し、再学習後に $c_0$ を上位に

---

<!-- _class: small -->
## 4.1 反復処理フロー（概要）

1. 初期ルールプール $\mathcal{P}$ と armプール $\mathcal{A}$ 準備（4.4）
2. arm選択：方策 $\pi$（UCB/ε-greedy/LLM-policy/Random）で $k$ 個選択（4.5）
3. acquire：サンプルしたターゲットに対し、Hornルール body を満たす evidence を候補グラフから探索（4.2）
4. evaluate：witness・新規evidence数などから代理報酬を計算し履歴更新（4.5）
5. KG更新：反復中に確定追加するのは **evidence のみ**、head は pending に保存

---

<!-- _class: small -->
## 4.2 Hornルールと証拠取得（acquire）

- Hornルール：

$$
 h: B(x,y,\mathbf{z})\Rightarrow (x,r^*,y)
$$

- body：

$$
 B(x,y,\mathbf{z})=(x,r_1,z_1)\wedge(z_1,r_2,y)\wedge\cdots
$$

- 重要：headを追加するのではなく、**body側の具体トリプル（evidence）を探索・取得**

---

<!-- _class: small -->
## 4.2 acquire手順（アルゴリズム視点）

1. サンプルしたターゲット $(x_0,r^*,y_0)$ を固定：$x\leftarrow x_0,\;y\leftarrow y_0$
2. 候補グラフ上で $B(x_0,y_0,\mathbf{z})$ を満たす代入 $\mathbf{z}$ を探索
3. 代入が見つかるたび、bodyを構成する具体トリプル集合を evidence として収集

- witness：bodyを満たす置換数（説明の厚み）
- NewEvidence：現KGに無かった evidence の本数

---

<!-- _class: small -->
## 4.2 Running example：ルールとevidence

例：

$$
 h_1:(x,\texttt{/people/person/place\_of\_birth},z)\wedge(z,\texttt{/location/location/containedby},y)\Rightarrow (x,r^*,y)
$$

- acquire では $(x,y)=(p_0,c_0)$ を固定
- 候補グラフから $z$（出生地）を探索
- 複数の $z$ が見つかれば witness 増
- body の具体トリプルが evidence（ただし $(p_0,r^*,c_0)$ は追加しない）

---

<!-- _class: small -->
## 4.3 なぜTAKGE（KG-FIT）か：直観

- 構造のみKGE（例：TransE）は、観測トリプルの近傍構造から学習
- 外部evidenceが「意味的に有益」でも、
  - ターゲット判別に必要な局所構造を補わない
  - ハブ化・偏り強化で誤帰納を誘発
  - 新規/疎なエンティティの表現が不安定
  などで精度向上に結びつかないことがある

---

<!-- _class: small -->
## 4.3 TransE：同型制約が平均化を誘発

- TransEスコア：

$$
 f_{\theta}(h,r,t)=-\lVert \mathbf{e}_h+\mathbf{r}-\mathbf{e}_t\rVert
$$

- 見通しの良い近似（平方損失）：

$$
 \min\sum_{(h,r,t)\in\mathcal{T}}\lVert \mathbf{e}_h+\mathbf{r}-\mathbf{e}_t\rVert^2
$$

- 同型パターンが大量追加：$(x_i,r_1,z)$, $(z,r_2,y_i)$

$$
 \mathbf{e}_z\approx\frac{1}{n}\sum_{i=1}^n(\mathbf{e}_{x_i}+\mathbf{r}_1)
$$

→ ハブ $z$ が平均（妥協）へ、周辺も集約 → 判別性低下

---

<!-- _class: xsmall -->
## 4.3 Running example（なぜTAKGEか：同型制約の例）

- Webで新規地名 $z$ が現れ、$(p_0,\texttt{/people/person/place\_of\_birth},z)$ と $(z,\texttt{/location/location/containedby},c_0)$ が追加
- さらに同様パターンが多人数に繰り返し追加：

$$
(p_1,\texttt{/people/person/place\_of\_birth},z),\;(z,\texttt{/location/location/containedby},c_0)\\
(p_2,\texttt{/people/person/place\_of\_birth},z),\;(z,\texttt{/location/location/containedby},c_0)\\
\cdots
$$

- 共通 $z$ を介した「同じ構造制約」が大量に増え、$z$ がハブ化
- 構造のみKGEでは平均化・集約が起き、$r^*$ 判別に効く差分が弱まる可能性
- TAKGEでは、$z$ のテキスト由来アンカー $\mathbf{a}_z$ と正則化で、疎ノードでも意味的に妥当な配置を促進

---

<!-- _class: small -->
## 4.3 TAKGE（KG-FIT）の学習目的

- 構造損失：

$$
\mathcal{L}_{\mathrm{KGE}}(\theta)=\sum_{(h,r,t)\in\mathcal{T}}\ell\big(f_\theta(h,r,t),\mathcal{N}(h,r,t)\big)
$$

- テキストアンカー（正規化）：

$$
\mathcal{L}_{\mathrm{anchor}}=\sum_{e\in\mathcal{E}}\lVert \hat{\mathbf{e}}_e-\hat{\mathbf{a}}_e\rVert^2
$$

- seed階層クラスタ中心 $\mathbf{c}_g$ による凝集＋分離：

$$
\mathcal{L}_{\mathrm{cohesion}}=\sum_{e}\lVert \hat{\mathbf{e}}_e-\hat{\mathbf{c}}_{g(e)}\rVert^2
$$

$$
\mathcal{L}_{\mathrm{sep}}=\sum_g\sum_{g'\in\mathcal{N}(g)}\max(0,\cos(\hat{\mathbf{c}}_g,\hat{\mathbf{c}}_{g'})-\tau)
$$

$$
\mathcal{L}_{\mathrm{TAKGE}}=\mathcal{L}_{\mathrm{KGE}}+\lambda_a\mathcal{L}_{\mathrm{anchor}}+\lambda_c\mathcal{L}_{\mathrm{cohesion}}+\lambda_s\mathcal{L}_{\mathrm{sep}}
$$

---

<!-- _class: small -->
## 4.4 初期ルールプールとarm生成

- ルールプール $\mathcal{P}$：AMIE+等で
  - support / head coverage / PCA confidence などの品質指標で上位を採用
- 探索単位：ルール単体ではなく「ルールの組」arm $a=\{h_{i_1},\dots,h_{i_m}\}$
- ねらい：相補性の取り込み
  - 同一仮説を異なる根拠で厚く支持（witness増）
  - 片方が拾えない evidence をもう一方が補う

---

<!-- _class: small -->
## 4.4 pair-arm：共起に基づく生成

- 候補集合 $\mathcal{S}_{\mathrm{cand}}$（ローカルKG由来）で、各ルールが支持し得るターゲット集合 $\mathcal{S}(h)$ を定義
- 共起スコア（Jaccard）：

$$
\mathrm{cooc}(h_i,h_j)=\frac{|\mathcal{S}(h_i)\cap\mathcal{S}(h_j)|}{|\mathcal{S}(h_i)\cup\mathcal{S}(h_j)|}
$$

- 上位ペアを pair-arm として採用

---

<!-- _class: small -->
## 4.5 arm選択：反復中は代理報酬

- 真の報酬：再学習後の評価改善だが、TAKGE再学習が高コスト
- 反復中は再学習せず、代理報酬で arm を評価・選択

$$
R(a)=\lambda_w\sum_{t\in\mathcal{T}_{\mathrm{sample}}}\mathrm{witness}(a,t)+\lambda_e\,|\mathrm{NewEvidence}(a)|
$$

- witness：説明の厚み（groundingの数）
- NewEvidence：現KGに無かった body トリプル本数

---

<!-- _class: small -->
## 4.5 witnessのprior重み（ハブ水増し対策）

- 関係ごとの prior $X_r$ により重み付け

$$
W(h)=\prod_{p\in\mathrm{body\_predicates}(h)}X_p
$$

- witness 集計：$\sum_{h\in a} W(h)c_h(t)$ の形で

---

<!-- _class: xsmall -->
## 4.5 Running example（報酬計算：grounding→witness→NewEvidence）

- arm $a=\{h_1,h_2\}$ を適用し、ターゲット $(p_0,r^*,c_0)$ に対して $h_1$ の body grounding を探索

$$
 h_1:(x,\texttt{/people/person/place\_of\_birth},z)\wedge(z,\texttt{/location/location/containedby},y)\Rightarrow(x,r^*,y)
$$

- 候補グラフ上で $z$ が2通り見つかった（grounding数=2）：

$$
 z=z_1:\;(p_0,\texttt{/people/person/place\_of\_birth},z_1),\;(z_1,\texttt{/location/location/containedby},c_0)\\
 z=z_2:\;(p_0,\texttt{/people/person/place\_of\_birth},z_2),\;(z_2,\texttt{/location/location/containedby},c_0)
$$

- witness（bodyを満たす置換数）：$\mathrm{witness}(a,(p_0,r^*,c_0))=2$
- NewEvidence（現KGに無かった body トリプル本数）：
  - 例：$(p_0,\texttt{/people/person/place\_of\_birth},z_1)$ は既知
  - $(z_1,\texttt{/location/location/containedby},c_0)$ と $(p_0,\texttt{/people/person/place\_of\_birth},z_2)$ が未観測
  - よって $|\mathrm{NewEvidence}(a)|=2$
- このターゲットでの代理報酬：$R(a)=\lambda_w\cdot 2+\lambda_e\cdot 2$（複数ターゲットなら合計）

---

<!-- _class: small -->
## 4.5.2 Arm選択方策（探索×活用）

- UCB / ε-greedy / LLM-policy（＋比較用Random）

### UCB

$$
\mathrm{UCB}(a)=\hat\mu(a)+\alpha\sqrt{\frac{\log t}{n(a)}}
$$

### ε-greedy
- 確率 $\varepsilon$ でランダム探索
- 確率 $1-\varepsilon$ で $\hat\mu(a)$ 最大（または上位$k$）を選択

---

<!-- _class: small -->
## 4.5.2 LLM-policy（意味整合で方策を補強）

- 統計だけでなく意味的整合性（semantic grounding）を評価して arm を選ぶ
- 入力情報（例）：
  - (i) ターゲット関係 $r^*$ の説明
  - (ii) arm内ルールの body/head predicate と自然言語説明
  - (iii) 直近反復の diagnostics（coverage / witness / overlap 等）
  - (iv) 取得・追加トリプル例（target/evidence/added）
- 出力：選択arm列＋更新後の方針テキスト（policy_text）
- ねらい：高報酬でも意味が薄いarmの抑制、意味整合の高いarmの重点化

---

<!-- _class: small -->
## 4.6 Webに基づく候補拡張（論文記述）

- ローカルKG限定だと新規エンティティ・周辺関係を取り込めない
- 各反復で Web 検索を併用し、LLM によって候補トリプル集合を外部から収集して候補グラフを拡張

手順：
1. 予算制約（問い合わせ回数・保持候補数の上限）
2. 候補抽出（bodyを満たし得る中間エンティティ＋必要候補トリプル、出典URL付与）
3. 正規化（表記ゆれ統合：決定論的ID付与、既存KGとの entity linking）
4. 漏洩防止（$r^*$ そのものを主張する候補は除外、body側補完に限定）
5. 保存と再現性（候補・出典を保存し監査可能に）

---

<!-- _class: small -->
## 4.6 Running example（Web候補）

- ローカルKGに $(z,\texttt{/location/location/containedby},c_0)$ が無い場合
- Webから「$z$ は $c_0$ の都市」と読める根拠を含む候補を収集
- 表記ゆれを正規化した上で候補集合へ加える
- ただし漏洩防止のため、$(p_0,r^*,c_0)$（国籍そのもの）を直接述べる候補は除外

---

<!-- _class: small -->
## 4.7 反復後の再学習と評価

- 反復で追加された evidence（必要に応じて新規エンティティ周辺の incident triples）を $\mathcal{T}_0$ に反映
- 更新KGで（TAKGE設定で）再学習し、最終評価（Hits@k / MRR）を算出
- 反復中は head を保留するため、改善は「周辺事実の増加の統合効果」として解釈できる

---

## まとめ（提案の要点）

- ターゲット関係 $r^*$ の改善を、外部根拠の統合で直接狙う
- Hornルールで探索空間を制御し、body側evidenceのみを確定追加（headは保留）
- 再学習が重いTAKGE環境でも、代理報酬で探索を回し、最後にまとめて再学習して評価
- Web候補統合では provenance・正規化・entity linking・漏洩防止を重視

---

<!-- _class: tiny -->
## 参考文献（ドラフト）

- [1] Bian. *LLM-empowered knowledge graph construction: A survey*. 2025.
- [2] Jiang et al. *KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge*. 2024.
- [3] Song et al. *MusKGC: A Flexible Multi-source Knowledge Enhancement Framework for Open-World Knowledge Graph Completion*. 2025.
- [4] Xiao et al. *DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion*. 2025.
- [5] Li et al. *Simple is Effective: The Roles of Graphs and Large Language Models in KG-RAG*. 2025.
- [6] Yang et al. *Embedding Entities and Relations for Learning and Inference in Knowledge Bases*. 2015.
- [7] Trouillon et al. *Complex Embeddings for Simple Link Prediction*. 2016.
- [8] Sun et al. *RotatE: Knowledge Graph Embedding by Relational Rotation*. 2019.
- [9] Paulheim. *Knowledge Graph Refinement: A Survey*. 2017.
- [10] Xue and Zou. *Knowledge Graph Quality Management: a Comprehensive Survey*. 2022.
- [11] Bordes et al. *Translating Embeddings for Modeling Multi-relational Data*. 2013.
- [12] Vrandečić and Krötzsch. *Wikidata: A Free Collaborative Knowledge Base*. 2014.
