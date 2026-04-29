# 目的（v3-combo-bandit への統合方針）

本派生（v3-combo-bandit）は「反復内では埋込スコア/ベクトルを各反復で計算しない」「候補トリプルの受理・報酬を witness と 衝突（conflict）に基づく代理指標で行う」設計である。
ここに、関係 (r) の **KGE-friendly（埋込モデルで表現しやすい）度合い**を表すスコア (X_r) を導入し、

1. witness を **信用度補正（重み付け）**して、ハブや無意味関係による witness 水増しを抑制
2. 初期arm構築（banditループ前）の **初期値（prior）**として利用し、supported-target Jaccard（cooc）だけでは拾えない「埋込的に良いルール/arm」を優先

する。

---

# 用語：X_r の位置づけ

* witness / evidence は arm 適用で実際に得られた置換（substitution）の数や支持構造
* (X_r) は witness そのものではなく「その witness をどれだけ信用すべきか」を与える **関係 r の事前スコア（prior / weight）**

以降、(X_r) は [0,1] のスカラーに正規化して扱う（未定義の場合は fallback）。

---

# X_r(7), X_r(2)〜X_r(4) の定義（型なしKG前提）

## (7) 幾何的一貫性（embedding space の変換の安定性）: X_r(7)

**前提**：初期のKGE（beforeモデル）で得られた entity embedding (\mathbf{e}_v) がある（反復内で再学習はしない。初期に一度だけ利用可能）。

**TransE/ベクトル差型の例（最も実装容易）**：

* 各トリプル ((a,r,b)) について (\delta=\mathbf{e}_b-\mathbf{e}_a) を計算
* 関係 (r) ごとの (\delta) の分散が小さいほど「同じ変換で説明できる」＝ KGE-friendly

例：
[
X_r^{(7)} = \exp!\left(-\mathrm{Var}_{(a,r,b)\in T_r}(\mathbf{e}_b-\mathbf{e}_a)\right)
]
（実装では分散の代わりに平均二乗偏差などでOK。最終的に [0,1] に正規化。）

**注意**：

* (T_r)（r のトリプル数）が少なすぎると不安定 → min_count を設け、未満なら X_r(7) を未定義扱い（fallback）。

---

## (2) hubness（ハブ汚染）: X_r(2)

目的：ハブ（高次数ノード）経由の witness 水増しを抑制する（ドキュメント中の `hub_bias` と整合）。

* 各 ((a,r,b)) について、tail 側（または両端）のグローバル次数 (\deg(\cdot)) を用いる
* ハブに寄るほどペナルティ

例（tail hub を見る）：
[
X_r^{(2)} = \frac{1}{\mathbb{E}_{(a,r,b)\in T_r}[\log(2+\deg(b))]}
]
または [0,1] に正規化して「大きいほど良い」に揃える。

---

## (3) 擬似型一貫性（type coherence の代替；型なしKG用）: X_r(3)

型情報が無いので、各エンティティの「どんな関係を持つか」を型 proxy として使う。

* 各エンティティ (v) について、述語分布ベクトル (S(v)) を作る（in/out の関係頻度）

  * (c^{out}_v(p)=|{u:(v,p,u)}|)
  * (c^{in}_v(p)=|{u:(u,p,v)}|)
  * (S(v)=\text{L1-normalize}([c^{out}*v(p)]*{p\in R}\Vert[c^{in}*v(p)]*{p\in R}))

関係 r の一貫性は、r の両端点の “役割” が揃っているかで測る：

[
X_r^{(3)}=\mathbb{E}_{(a,r,b)\in T_r}[\cos(S(a),S(b))]
]

（型の代わりに role coherence を測る。高いほど schema-like / KGE-friendly。）

計算が重ければ、頻出関係のみ・上位K述語に制限などで近似可。

---

## (4) パターン性 / 低自由度（低ランクっぽさのproxy）: X_r(4)

型不要。関係 r の “集中度（1対1/多対1寄り）” や “拡散の小ささ” を測る。

実装が簡単な proxy（1対1/機能的衝突の抑制に効く）：

[
X_r^{(4)}=\mathbb{E}_{a}\frac{1}{1+|N^{out}*r(a)|} ;+; \mathbb{E}*{b}\frac{1}{1+|N^{in}_r(b)|}
]

* (N^{out}_r(a)={b:(a,r,b)\in T})
* (N^{in}_r(b)={a:(a,r,b)\in T})
* 拡散が大きい（候補が増えやすい）関係は小さくなる

---

# X_r の統合（最終的に使うスコア）

各 r について、上記を [0,1] に正規化してから統合する。

推奨（安定運用）：

* 主軸：X_r(7)（あれば使う）
* ガード：X_r(2)（ハブ汚染抑制）
* 補助：X_r(3), X_r(4)（型なしの安定化・拡散抑制）

例（ゲーティング + fallback）：

* まず hubness と拡散で足切り（または強い減衰）
* その上で X_r(7) を優先、無ければ (3)/(4) を使う

擬似コード例：

```python
if not reliable(X7[r]):        # few triples etc.
    base = 0.5*X3[r] + 0.5*X4[r]
else:
    base = X7[r]

X[r] = base * X2[r]            # hubnessで減衰
X[r] = clamp01(X[r])
```

（運用上は「いきなり hard filter を強くしない」ため、まずは減衰で十分。）

---

# 使い方1：witness の重み（代理指標の改良）

現行設計では候補 ((x,r_*,y)) に対して arm 内ルールの witness を用いて支持を加点する（例：`q(x,y)=log(1+sum c_h(x,y))`）。
ここを **関係 r の信用度 (X_r)** で重み付けして、無意味関係やハブ経由の witness 水増しを抑える。

## ルール重み W(h)

body=2 ルール (h) の body に含まれる関係を (r_1, r_2) とする。

* ルール重み：
  [
  W(h)=X_{r_1}\cdot X_{r_2}
  ]
  （または幾何平均でもOK）

## 候補スコア q(x,y) の変更

[
q(x,y)=\log\left(1+\sum_{h\in a} W(h),c_h(x,y)\right)
]

## arm 報酬 R(a) の変更（例）

ドキュメントの形を維持しつつ、q を差し替える：
[
R(a)=\frac{1}{|C_a^{acc}|}\sum_{(x,y)\in C_a^{acc}} q(x,y);-;\lambda,\mathrm{conflicts}(C_a^{acc});-;\mu,\mathrm{hub_bias}(C_a)
]
（`hub_bias` は既存のままでよいが、X_r(2) を導入することで hub 由来 witness が減り相補的。）

---

# 使い方2：初期arm選択の初期値（prior）として使う

現行の初期arm生成は supported-target Jaccard（`metadata.cooc`）で pair-arm をランキングし、singleton も作る。
これに (X_r) を導入し、cooc が同程度なら **KGE-friendly なルール/arm を優先**する。

## ルールの KGE-friendly スコア（prior）

body=2 ルール (h) の prior を
[
\pi(h)=W(h)=X_{r_1}X_{r_2}
]
とする（上と同じ定義でよい）。

## pair-arm の seed スコア（cooc × prior）

pair-arm ({h_i,h_j}) のランキングを、現行の cooc（Jaccard）を保ちつつ prior を掛ける：

[
\text{seed}(h_i,h_j)=\mathrm{cooc}(h_i,h_j)\cdot \sqrt{\pi(h_i)\pi(h_j)}
]

* `metadata.cooc` はそのまま保持
* 追加で `metadata.prior` や `metadata.seed_score` を保存してもよい

これにより、

* cooc が高いがハブ水増し（X2低）や幾何が不安定（X7低）のペアは下がる
* cooc が中程度でも KGE-friendly なペアが上がる

## UCB/ε-greedy の初期平均報酬としての利用（任意）

bandit ループ開始時に履歴が無いので、arm の初期値として prior を使う（実装しやすい）。

例：

* `ArmHistory` 初期化時に `estimated_mean_reward = prior(a)` を入れる
* `prior(a)` は arm 内ルール prior の平均/合計などで定義

[
\text{prior}(a)=\frac{1}{|a|}\sum_{h\in a}\pi(h)
]

（UCB の “初期値” として扱う。探索性を壊さないよう、あくまで初期のみ。）

---

# 実装メモ（どこに入れるか）

* X_r 計算（オフライン、初期構築時）

  * 新規：`compute_relation_Xr.py` 等で `relation_priors.json` を出力（r -> X2,X3,X4,X7,X）
  * X7 は before KGE モデルの entity embeddings を読み込んで計算（反復内では計算しない）

* 初期arm生成

  * `simple_active_refine/arm_builder.py`
  * pair-arm のランキング（cooc降順）を `seed_score` 降順に変更 or tie-break に prior を使用

* witness重み付け

  * `simple_active_refine/triple_evaluator_impl.py`（q(x,y) を計算している箇所）
  * `count_witnesses_for_head` で得た c_h(x,y) に対し W(h) を掛ける
  * 既存の衝突（functional conflict）・hub_bias のロジックは維持
