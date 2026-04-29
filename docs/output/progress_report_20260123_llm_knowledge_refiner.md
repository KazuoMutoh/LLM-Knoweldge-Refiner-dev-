# 進捗報告（2026-01-23）: LLM Knowledge Refiner / ルール駆動KG精錬の研究

作成日: 2026-01-23  
対象: 大学向け進捗報告（アルゴリズム中心、実装詳細は省略）

---

## 1. 研究の狙い（何を解決したいか）

知識グラフ（KG）に対して、外部情報源や候補集合から新しいトリプル（事実）を追加し、
最終的にリンク予測性能（MRR/Hits）や、特定の目的関係（例: 国籍）の推定品質を改善したい。

ただし、単純に「候補トリプルを大量に追加する」だけでは、
- 追加がノイズになり埋め込み学習（KGE）が悪化
- ターゲット関係の推定がむしろ下がる

といった現象が起こり得ることが、初期実験で確認された。

本研究では、以下の問いに段階的に答える形で研究を進めている。

- Q1: どのトリプルを追加すれば、ターゲット関係（例: nationality）の推定が改善するのか？
- Q2: 「追加すべき候補」を、KGEを毎回再学習せずに（計算を抑えつつ）選別できるか？
- Q3: 構造のみKGEで起きる悪化を、テキスト属性（意味）を使って緩和できるか？
- Q4: ローカル候補集合（train_removed）以外に、Webから根拠トリプルを取得して精錬できるか？

---

## 2. 全体アプローチの整理（研究で扱う要素）

本研究は「ルール（Hornルール）に基づいて候補を生成し、反復的にKGを更新し、最終的に再学習で評価する」枠組みを採用している。

大枠は次の3要素の組合せで設計・実験を進めてきた。

1. **ルール抽出（初期プール）**
   - KGから高信頼部分を抽出し、AMIE+ 等で Horn ルールを得る
   - ルールは「なぜそのターゲットが成立しそうか」を構造的に記述する

2. **反復精錬（追加トリプルの選別・追加）**
   - ルール（またはルールの組合せ）を選び、候補集合から根拠・周辺トリプルを追加
   - 反復内でKGEを再学習しない（計算資源節約）前提で、代理指標で良さそうな選択を行う

3. **最終評価（再学習・指標評価）**
   - 反復で追加されたトリプルを train に反映し、KGEを再学習して評価する

評価軸は2本立てで運用している。

- **ターゲット最適化指標（主）**: ターゲットトリプルのスコア改善（特にモデル間比較のため minmax(train) 正規化を利用）
- **一般性能指標（副）**: Hits@k / MRR（リンク予測の標準指標）

---

## 3. 最新アルゴリズムの詳細（2026-01-23時点）

この節では「現時点で実験に投入している最新の標準アルゴリズム」を、
入力/出力、反復ループ、評価まで含めて整理する。

本プロジェクトの最新の位置づけは、次の組合せである。

- **反復精錬（arm-based / combo-bandit）**: ルールの組合せ（arm）を選び、候補集合から根拠（evidence）と周辺（incident）を追加する
- **選択戦略**: UCB または LLM-policy（直近の強い結果は UCB）
- **KGE（再学習・再スコア）**: TAKG前提の KG-FIT（TransE / PairRE）

参照（設計標準）:
- RULE-20260111-COMBO_BANDIT_OVERVIEW-001
- RULE-20260119-TAKG_KGFIT-001
- RULE-20260118-RELATION_PRIORS-001（必要に応じて）

### 3.1 問題設定と記法

- 目的関係（target relation）を $r_*$ とする（例: nationality）。
- ターゲット集合（説明・改善したい事実）を $\mathcal{T} = \{(h, r_*, t)\}$ とする。
- 現在の知識グラフ（学習用トリプル集合）を $G_k$（反復 $k$ の時点）とする。
- Hornルール集合（初期プール）を $\mathcal{P}$ とし、arm を「ルールの組合せ」として $\mathcal{A}$ とする。

本研究の設計上の重要点は、**反復ループ内ではKGEを再学習しない**ことにある。
（KGEの再学習は最終段でまとめて行い、反復中は代理指標で探索する。）

### 3.2 入力と出力（研究で扱う実体）

入力（固定・実験条件として与えるもの）:
- dataset_dir（train/valid/test、必要に応じてTAKG用テキストやKG-FIT成果物）
- $r_*$ と target_triples（$\mathcal{T}$）
- 候補集合 $C$（典型: train_removed.txt。将来: Web取得）
- before model（KGE; スコア計算・priors算出・評価の基準）
- 反復設定（反復回数、arm数、1反復で選ぶarm数、1armあたりの対象target数など）

出力（反復の成果物）:
- 各反復で受理された追加トリプル集合（accepted_added_triples；重複除去後）
- arm履歴（どのarmを選び、どれだけの根拠を得て、どの程度の報酬だったか）
- 最終的な updated dataset（train_after など）
- after model（再学習）と評価指標（target / Hits / MRR）

平易な言い換え:
- 入力は「今あるKG」「改善したいターゲット例」「探す場所（候補集合 or Web）」「実験の設定値」。
- 出力は「どんな根拠（追加トリプル）を集めたか」「どの作戦（arm）が良かったかの記録」「追加後に学習し直した結果」。

### 3.3 準備ステップ（反復前のオフライン処理）

#### (A) 初期ルールプール $\mathcal{P}$ の構築

狙い:
- 候補集合 $C$ を「無差別に追加」するのではなく、ターゲットに関係しそうな構造パターン（Hornルール）を抽出して、探索空間を圧縮する。

概略:
1. before KG（train）でKGEを学習（または既存モデルをロード）
2. 高信頼トリプル近傍から AMIE+ で Horn ルールを抽出
3. 品質指標（head coverage / PCA confidence など）でフィルタし、プールとして固定

（設計標準: RULE-20260111-KG_REFINE_V3_OVERVIEW-001）

平易な言い換え:
- いきなり候補を全部見ると広すぎるので、「こういう形の証拠があれば国籍が分かりやすい」という“型（ルール）”を先に集めて、探索範囲を絞る。

#### (B) 初期arm集合 $\mathcal{A}$ の構築（combo-banditの要点）

狙い:
- 「単一ルール」ではなく「複数ルールの組合せ」で、同一ターゲットを多角的に支持できる可能性がある。

概略:
- armは singleton（1ルール）と pair（2ルール）を主に使う。
- pairの候補は、ターゲット集合 $\mathcal{T}$ に対して「支持対象の重なり（cooc）」が大きい組を優先する。
  - 直観: 同じターゲットを複数ルールで説明できるなら、支持構造（witness）が厚くなる可能性が高い。

（設計標準: RULE-20260111-COMBO_BANDIT_OVERVIEW-001）

平易な言い換え:
- ルールは1本よりも「組合せ」で効くことがある。
- そこで「どのルールセットを使って証拠集めをするか」を“腕（arm）”として用意し、後で良い腕を学習的に見つける。

#### (C) relation priors（任意）

狙い:
- witness がハブ関係で水増しされる問題を抑制し、反復中の代理報酬を「KGEにとってフレンドな根拠」に寄せる。

要点:
- relationごとの prior $X_r\in[0,1]$ をオフライン算出し、witness を重み付けする。
- ただし直近の強い比較（UCB vs Random）は **priors=off** で実施し、混入要因を避けた。

（設計標準: RULE-20260118-RELATION_PRIORS-001）

平易な言い換え:
- 証拠の数だけを見ると、よく出る関係（ハブ）があるだけで“効いている風”に見えてしまう。
- そこで「この関係は根拠として信頼しやすい/しにくい」という重みを先に作って、証拠の“質”を調整する。

#### (D) KG-FIT（TAKG）成果物（任意だが現状は重要）

狙い:
- 構造のみKGEでは、追加トリプルがノイズになってターゲットを悪化させるケースがある。
- テキスト属性（意味アンカー）と階層制約をKGE学習に統合し、悪化を緩和する。

要点:
- entityテキスト埋め込み（name/desc）と seed階層（クラスタ）を事前計算し、KG-FIT学習で正則化として利用する。
- interaction modelとして TransE/PairRE を選び、条件比較を行う。

【数式（KG-FITの目的関数イメージ）】

KG-FITはリンク予測損失（構造）に、テキスト・階層に基づく正則化項を加える。
概略として
$$
\mathcal{L}=\mathcal{L}_{\text{link}}+\lambda_{\text{anc}}\,\mathcal{L}_{\text{anchor}}+\lambda_{\text{coh}}\,\mathcal{L}_{\text{cohesion}}+\lambda_{\text{sep}}\,\mathcal{L}_{\text{separation}}
$$
のように書ける。

- $\mathcal{L}_{\text{link}}$: 通常のKGE（TransE/PairRE等）のリンク予測損失
- $\mathcal{L}_{\text{anchor}}$: entity埋め込みがテキスト埋め込み（意味アンカー）から大きく逸脱しないようにする項
- $\mathcal{L}_{\text{cohesion}}$: 同一クラスタ（seed階層）内で近づくようにする項
- $\mathcal{L}_{\text{separation}}$: 近傍クラスタとは margin をもって分離する項

直観:
- 追加トリプルが局所的に強い制約になっても、意味アンカーと階層制約により「全体として無理のない配置」へ戻す力を入れる。

（設計標準: RULE-20260119-TAKG_KGFIT-001 / RULE-20260123-KGFIT_PAIRRE-001）

平易な言い換え:
- 構造だけで学習するKGEは、トリプルを足すと「無理に全体を合わせようとして」ターゲットが崩れることがある。
- KG-FITは、エンティティのテキスト説明（意味）を“錨（いかり）”にして、追加で揺れても意味的に変な位置へ行きすぎないようにする発想。

### 3.4 反復ループ（arm-driven refinement）

反復 $k=1..N$ の各ステップは次の通り。

#### Step 1: arm選択（探索と活用）

目的:
- 限られた反復回数で、有望なarmへ試行回数を配分する。

代表例:
- **UCB**: 平均報酬 + 未試行ボーナスで選択（探索/活用のバランス）
- **LLM-policy**: armの意味（関係の説明）や直近の取得例を与えて、ターゲット文脈に整合するarmを選ばせる

直近の重要知見は「UCBが安定しやすい」傾向である（後述のUCB vs Random で顕著）。

平易な言い換え:
- 反復回数には限りがあるので、「当たりそうな作戦を多めに試す（活用）」と「まだ試していない作戦も少し試す（探索）」のバランスを取る。
- UCBは、このバランスを数式で自動化した選び方。

【数式（UCBの典型形）】

arm $a$ の $k$ 回目時点の選択スコアを
$$
\mathrm{UCB}_k(a)=\hat\mu_k(a) + c\sqrt{\frac{\ln k}{n_k(a)}}
$$
とする。

- $\hat\mu_k(a)$: これまでの報酬の平均（proxy reward）
- $n_k(a)$: arm $a$ の試行回数
- $c>0$: 探索の強さ

直観:
- $\hat\mu$ が大きいarmは活用されやすい。
- $n(a)$ が小さいarmは「未探索ボーナス」で試されやすい。

【具体例（UCBの挙動イメージ）】

反復 $k=50$、$c=1$ とし、
2つのarmについて

- $a_1$: $\hat\mu=10$, $n=25$
- $a_2$: $\hat\mu=8$, $n=1$

なら
$$
\mathrm{UCB}(a_1)\approx 10 + \sqrt{\ln 50 / 25}\approx 10.40,
$$
$$
\mathrm{UCB}(a_2)\approx 8 + \sqrt{\ln 50 / 1}\approx 9.98
$$
で、平均は $a_1$ が高いが、未試行の $a_2$ も一定確率で試される。

#### Step 2: 候補取得（local候補集合から evidence を集める）

基本設定（現状の標準）:
- 候補集合 $C$ は train_removed.txt を用いる（ローカル候補）。
- 各armについて、ターゲット集合 $\mathcal{T}$ から少数サンプルし、ルールbodyを満たす「根拠トリプル（evidence）」を $C$ 内から探索する。

ここで重要なのは、生成される追加が2種類ある点である。

1) **evidence triples**: ルールbodyを満たすために必要なトリプル
2) **incident triples**: 中間ノード（?c 等）周辺の、説明構造を補うトリプル（上限やON/OFFで制御可能）

平易な言い換え:
- ここは「証拠集め」の工程。
- ルールが要求する“つながり”を、候補集合の中から探して持ってくる。
- 直接の証拠（evidence）に加えて、説明に必要な周辺情報（incident）も少し足すと、後の学習で意味が通りやすくなることがある。

【数式（Hornルールと変数代入）】

Hornルール $h$ を
$$
h: \; B_1(\cdot)\wedge B_2(\cdot)\wedge\cdots\wedge B_m(\cdot) \Rightarrow (x,r_*,y)
$$
とする。

ターゲット $t=(h,r_*,t)$ が与えられたとき、headの形 $ (x,r_*,y)$ と $t$ を単一化（unification）して
初期代入 $\theta_0$（例: $\theta_0(x)=h,\theta_0(y)=t$）を得る。

次に、候補集合 $C$ 上で body が満たされる代入 $\theta\supseteq\theta_0$ が存在すれば、
そのルールはターゲットを支持する（supports）。

【具体例（nationalityの典型パターン）】

ターゲット関係を $r_*=$ `/people/person/nationality` とする。

次のようなbody=2ルールを考える。

$$
h_\text{born}:\; (x,\text{place\_of\_birth},c)\wedge(c,\text{contained\_by},y)\Rightarrow (x,\text{nationality},y)
$$

ターゲット $t=(\text{BarackObama},\text{nationality},\text{USA})$ に対して $\theta_0(x)=\text{BarackObama},\theta_0(y)=\text{USA}$。

候補集合 $C$（train_removed など）に

- $(\text{BarackObama},\text{place\_of\_birth},\text{Honolulu})$
- $(\text{Honolulu},\text{contained\_by},\text{USA})$

が存在すれば、$\theta(c)=\text{Honolulu}$ により body が満たされ、
ルール $h_\text{born}$ はターゲットを支持する。

このとき追加されるのは、
- evidence: 上の2本（bodyを満たす根拠）
- incident: 例えば $(\text{Honolulu},\text{located\_in},\text{Hawaii})$ など、説明構造を補う周辺トリプル
であり、ターゲットそのもの（nationality）は「確定追加」ではなく、反復中は支持構造を厚くするために周辺を集める設計になっている。

#### Step 3: 代理報酬の計算（witness / conflict）

反復中はKGEを回さないため、armの良さは代理指標で評価する。

現状の中心は次の2系統である。

- **witness（支持構造）**: どれだけ多くの置換（body成立）でターゲットを支持できるか
  - priorsを使う場合は、body predicate の prior を掛け合わせて重み付けする
- **conflict（機能的矛盾）**: 1対1寄りの関係（例: nationality）に対して矛盾する候補が増えすぎていないか

狙い:
- witness は「説明の厚み」を表し、conflict は「矛盾の混入」を抑える安全弁になる。

平易な言い換え:
- 反復中は毎回学習し直せないので、「この作戦は良かった/悪かった」を代わりの尺度で判断する。
- witness は“同じ結論を支える根拠がどれだけ集まったか”。
- conflict は“矛盾する結論が増えすぎていないか”。
- つまり「根拠を厚くしつつ、矛盾は増やさない」方向へ探索を誘導する。

【数式（witnessの定義）】

ルール $h$ とターゲット $t$ に対し、bodyを満たす代入（置換）集合を
$$
\Omega(h,t)=\{\theta\mid \theta\supseteq\theta_0(t,h),\; C\models B\,\theta\}
$$
と置くと、witness数は
$$
c_h(t)=|\Omega(h,t)|
$$
で定義できる。

arm $a$ が複数ルールの集合であるとき、素朴には
$$
\mathrm{witness}(a,t)=\sum_{h\in a} c_h(t)
$$
で「支持の厚み」を表せる。

【数式（relation priorsによる重み付け）】

bodyに含まれる述語集合を $\mathrm{pred}(h)$ として、
$$
W(h)=\prod_{p\in\mathrm{pred}(h)} X_p
$$
とおけば、重み付きwitnessは
$$
\mathrm{witness}_X(a,t)=\sum_{h\in a} W(h)\,c_h(t)
$$
となる。

直観:
- ハブ関係やKGE非フレンドな関係（$X_p$ が小さい）がbodyに入ると、その支持は割り引かれる。

【数式（conflictの一例: 機能的関係の矛盾）】

nationalityのように「1人に1国籍寄り」と見なす場合、head $x$ に対して複数の $y$ が強く支持される状況は矛盾候補になる。

例えば、候補として集まった head-triple（推論候補）集合を $H_x=\{y\mid (x,r_*,y) \text{が支持される}\}$ とし、
$$
\mathrm{conflict}(x)=\max(0,|H_x|-1)
$$
のような単純ペナルティを置ける。

より実務的には、上位K候補に限定する、支持スコアが閾値を超えたものだけ数える等の工夫を入れる。

【具体例（conflictが増えるケース）】

同一人物 $x$ に対して、
- $y=\text{USA}$ を支持する証拠（出生地→包含）
- $y=\text{Kenya}$ を支持する証拠（出生地→包含）

が同程度に多く集まると、$|H_x|=2$ となり conflict が発生する。
このような矛盾候補を抑えるため、conflict を報酬から減点する（あるいは受理段階で弾く）設計が必要になる。

【数式（proxy rewardの典型形）】

arm $a$ の反復 $k$ における報酬 $R_k(a)$ は、例えば
$$
R_k(a)=\alpha\sum_{t\in\mathcal{T}_k} \mathrm{witness}_X(a,t)\; -\; \beta\sum_{x\in\mathcal{H}_k} \mathrm{conflict}(x)\; +\; \gamma\,|E_k(a)|
$$
のように置ける。

- $\mathcal{T}_k$: その反復でサンプルしたターゲット集合
- $\mathcal{H}_k$: その反復で登場した head（人物など）
- $E_k(a)$: そのarmで得られた（受理された）evidence数
- $\alpha,\beta,\gamma$: 重み

重要なのは、反復中は「KGEスコア改善」ではなく、説明の厚みと矛盾抑制で探索を回す点である。

#### Step 4: KG更新と履歴更新

- 受理された追加トリプルを $G_{k-1}$ に加えて $G_k$ を得る。
- 同時に、armの報酬・取得量・診断指標を履歴として保存し、次反復の選択に使う。

平易な言い換え:
- ここは「集めた証拠をKGに足す」「今回の作戦の成績を記録する」。
- 記録が溜まると、次の反復で“当たりやすい作戦”が見つけやすくなる。

### 3.5 最終ステップ（after再学習と評価）

反復で得た追加トリプル集合（重複除去後）を train に反映して updated dataset を作り、
KGEを再学習して before/after を比較する。

評価は2軸（ターゲット最適化 / 一般性能）で行う。

1) **ターゲット指標（主）**
- target_triples を after model で再スコアし、beforeとの差を見る。
- モデル間比較では minmax(train) 正規化を用いる。
- unknown によりスコア不可になるターゲットがあるため、
  「全モデルで共通にスコア可能な集合（coverageの共通部分）」で比較する。

2) **一般指標（副）**
- Hits@k / MRR を test set で測る。

この評価設計により、
- ターゲット改善はしているが一般指標が悪化している
- 逆に一般指標は改善しているがターゲットは改善していない

といったトレードオフ構造を切り分けて議論できる。

平易な言い換え:
- 反復で集めた証拠は「このままでは仮説」なので、最後にまとめて学習し直して、本当に良くなったかを確かめる。
- そのとき、(i) ターゲットだけ良くなったのか、(ii) 全体のランキングも良くなったのか、を分けて評価する。

【数式（minmax(train) 正規化）】

afterモデル（またはbeforeモデル）での raw score を $s(t)$ とし、
そのモデルの train スコアの最小・最大を $s_{\min}^{\mathrm{train}}, s_{\max}^{\mathrm{train}}$ とする。
min-max 正規化は
$$
	ilde{s}(t)=\frac{s(t)-s_{\min}^{\mathrm{train}}}{s_{\max}^{\mathrm{train}}-s_{\min}^{\mathrm{train}}}
$$
で与える。

ターゲット改善量は
$$
\Delta(t)=\tilde{s}_{\mathrm{after}}(t)-\tilde{s}_{\mathrm{before}}(t)
$$
で定義し、平均改善 $\Delta_{\mathrm{mean}}$ や改善率（$\Delta(t)>0$ の割合）を集計する。

【具体例（正規化が必要になる理由）】

KGEの種類（TransE vs KG-FIT(PairRE)）や設定により raw score スケールが異なる。
例えば、
- beforeモデルA: trainのraw scoreがだいたい $[-20,-5]$
- afterモデルB: trainのraw scoreがだいたい $[-3000,-2000]$

のように桁が違うと、rawの差分は比較不能になる。
このとき、各モデルの train 分布で 0..1 に正規化してから比較することで、
「そのモデルの中でターゲットが相対的に上がったか」を同じ物差しで測れる。

【coverage（共通集合）を取る理由】

target_triples の一部が unknown としてスコア不可になる場合があるため、
比較では
$$
\mathcal{T}_{\cap}=\mathcal{T}_{\mathrm{before}}^{\mathrm{scorable}}\cap\mathcal{T}_{\mathrm{after}}^{\mathrm{scorable}}
$$
のように「両方でスコア可能」な共通集合で集計する。

---

## 4. 最初から何が変わったか（アルゴリズムの変遷と意思決定）

以下では、docs/database/index.md に登録されている records / rules を軸に、
「どんな課題に直面し、どの実験で何を確かめ、どう改善したか」を時系列でまとめる。

### 4.0 変遷の全体像（要約）

最初期から現在までの主な変更点を「課題→打ち手→狙い」で並べる。

| 段階 | 直面した課題 | 導入した打ち手（変更点） | 狙い |
|---|---|---|---|
| (0) ベースライン | 候補を足すと良くなる？が不明 | add-all / ランダム追加など | まず前提の検証 |
| (1) v3 | 無差別追加は探索空間が広すぎる | Hornルール抽出で探索空間を圧縮 | ターゲットに関連する構造へ誘導 |
| (2) arm（combo-bandit） | 単一ルールでは効きが弱い/不安定 | ルールの組合せ（arm）を探索単位に変更 | 多角的支持（witness）を得る |
| (3) 代理指標の整備 | 反復中にKGEを回せない | witness/conflict を中心に proxy reward 化 | 低コストで探索を回す |
| (4) priors | witness がハブ関係で水増し | relation priors で witness を重み付け | KGEフレンドな根拠へ寄せる |
| (5) TAKG/KG-FIT | 追加で構造KGEが悪化する | テキスト属性+階層正則化（KG-FIT）導入 | 悪化緩和・意味アンカー導入 |
| (6) PairRE / 比較 | targetとMRRがトレードオフ | PairREなど表現力比較、正規化再スコア | 目的指標に対する妥当な比較 |
| (7) Random baseline | 「選別の効果」の根拠が必要 | 同数ランダム追加との比較 | UCB選別の有意性を示す |

### 4.1 v3（ルール駆動）: ルール抽出 → 反復追加 → 再学習

- 基本枠組み（設計標準）
  - RULE-20260111-KG_REFINE_V3_OVERVIEW-001

狙い:
- Hornルールにより「ターゲットに効く周辺構造」を明示し、
  ルールに沿ったトリプル追加で改善を狙う。

観測された課題:
- 反復で追加するトリプルが、必ずしもターゲット改善に繋がらない。
- 追加の副作用として、KGEの学習が悪化しうる（特に候補集合の全投入など）。

### 4.2 combo-bandit（arm運用）: 「ルールの組合せ」を腕として探索

- arm（ルール組）を選択単位にする派生（設計標準）
  - RULE-20260111-COMBO_BANDIT_OVERVIEW-001

狙い:
- 個々のルールより「組合せ」が効く可能性がある。
- 探索・活用（exploitation/exploration）を、バンディット的に制御して、限られた反復回数で良いarmを見つけたい。

重要な設計判断:
- 反復中にKGEを再学習せず、代理指標で報酬（reward）を付与して探索を進める。

### 4.3 代理指標の改善: witness の水増し問題と relation priors

- 背景: witness（説明の厚み）を、反復内の主要な代理指標として使うと、
  ハブ的な関係（頻出関係）やKGEで扱いにくい関係により witness が「水増し」される。

- 改善策: relation priors（関係ごとの事前スコア）で witness を重み付けする
  - RULE-20260118-RELATION_PRIORS-001
  - （運用の実験結果集約）REC-20260118-FULL_PIPELINE_RESULTS-001

効果（暫定の理解）:
- LLM-policy と UCB の挙動差、追加量、指標変化を比較できるようになり、
  少なくとも本データでは「UCBの方が安定」な傾向が観測された。
- ただし、priors導入のみでターゲット改善が自動的に得られる、という単純な結論にはなっていない。

### 4.4 「追加しすぎると悪化する」ことの確認（ベースライン）

- 課題: そもそも候補集合（train_removed）を追加する行為自体が、ターゲットを悪化させる可能性。

- 検証: train_removed を全投入（add-all）した場合の before/after
  - REC-20260118-ADD_ALL_TRAIN_REMOVED-001

要点:
- 全投入で target score も Hits/MRR も悪化するケースを確認。
- したがって「候補集合を全部足せば改善」という前提は成立しない。
- ここから「何を追加するか（選別）」が必須である、という方向が明確化した。

### 4.4.1 データセット定義と評価の妥当性（混入要因の整理）

背景:
- KG精錬の実験は、データ生成手順（train/valid/test の作り方、target_triples の扱い）に強く依存する。
- 「add-all が悪化する／改善する」など、一見矛盾する結果が出た場合、
  追加手法の良し悪し以前に **実験セットアップの差**が混入している可能性がある。

対応:
- データ生成（make_test_dataset）の仕様を明確化し、再現性のある形でデータセットを作れるようにした上で、
  drop_ratio=1.0（文脈削除） vs add-all（候補全投入）を再検証した。

結果（修正版データセットでの再検証）:
- TransE / KG-FIT の両方で、minmax 正規化 target スコアにおいて add-all が有意に改善（ただし改善幅はモデルで異なる）。

参照:
- REC-20260120-DROP_RATIO_1_ADD_ALL-003
- REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-003

### 4.5 TAKG / KG-FIT の導入: 構造のみKGEの悪化をテキスト属性で緩和

- 背景: 構造のみKGEは、追加トリプルにより局所的に過制約・ノイズ影響を受けやすい。

- 対応: テキスト埋め込みと階層制約（seed階層）をKGEの学習に統合する KG-FIT を採用
  - RULE-20260119-TAKG_KGFIT-001

- 既存の追加前/追加後KGを使って、KGEをKG-FITへ置換して再評価
  - REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001

要点（現時点の整理）:
- KG-FITに置き換えるだけで「常に悪化が止まる」とは言い切れない。
- ただし、条件によっては悪化幅が小さいケースも観測され、
  以降は「設定スイープ（正則化強度、neighbor_k等）」が重要な課題になった。

---

## 4.x 補足: 評価設計上の重要ポイント（現在の標準運用）

本研究では、アルゴリズム比較の公平性のために、評価手順も段階的に整備してきた。

1) **スコアの正規化**
- モデル間で raw score スケールが変わるため、ターゲット比較では minmax(train) を主に用いる。
- sigmoid 正規化は飽和して差分が潰れることがあるため、用途を限定する。

2) **ターゲット coverage（unknown 除外）の明記**
- before/after やモデル種別によって、target_triples の一部が unknown 扱いになることがある。
- 比較では「全モデルで共通にスコア可能な集合」を用い、除外数も併記する。

これらは「手法の当たり外れ」ではなく、研究としての主張を成立させるための基盤（再現性・妥当性）として重要である。

---

## 5. 直近の主要成果: KG-FIT(PairRE) + UCB がランダム追加を明確に上回る

ここが現在の最も強い進捗ポイントである。

### 5.1 実験の問い

- 同じ候補集合（train_removed）から、
  - UCB（arm-runで選ばれた追加）
  - ランダム（同数を無作為抽出）

で、ターゲット改善に差が出るか？

### 5.2 比較の前提（公正性のための工夫）

- 主指標は **minmax(train) による正規化ターゲット再スコア**
  - raw score はモデルによりスケールが異なるため、比較が不安定になり得る
- unknown entity/relation によりスコア不可のターゲットがあるため、
  **全モデルで共通にスコア可能なターゲット集合**で比較

### 5.3 結果（UCB vs Random）

- UCB条件の基礎（KG-FIT(PairRE), arm=10, priors=off）
  - REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001

- ランダム追加ベースライン（同数サンプリング、反復seed）
  - REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001

要点（seed0..3の範囲の暫定まとめ）:
- 比較対象: 共通でスコア可能なターゲット 106本
- UCB（KG-FIT(PairRE)）:
  - minmax(train) の Δmean = **+0.0412**
  - improved_frac = **0.9245**
- Random（N=10232, seed0..3）:
  - Δmean は全seedで負（平均 **-0.153**、標準偏差 約 **0.016**）
  - improved_frac は全seedで **0**

解釈:
- 「同数をランダムに追加する」だけではターゲットが一貫して悪化する一方、
  **UCB（arm-runが選別した追加）はターゲット改善を一貫して引き出している**。
- したがって、少なくとも本条件では「選別（bandit選択）に意味がある」ことが実験的に示せた。

注意（トレードオフ）:
- 同じ実験群で、Hits/MRR が必ずしも同方向に改善するとは限らない。
- 今後は「ターゲット最適化」と「一般指標」の両立条件探索が課題。

---

## 6. 研究としての学び（現時点の結論）

現段階で、研究として明確になったことを整理する。

1. **候補集合を無差別に追加すると悪化する**
   - add-all（全投入）でも悪化するケースがあるため、「追加 = 改善」の単純仮説は棄却。

2. **反復内でKGEを再学習しなくても、選別は可能性がある**
   - arm選択（UCB）とルール構造に基づく追加が、ランダムより明確に良い結果を出した。

3. **ターゲット指標と一般指標はトレードオフになり得る**
   - ターゲットを改善する設定が、MRR/Hitsを悪化させることがある。
   - 評価軸（研究目的）を明確にし、両立条件を探索する必要がある。

4. **TAKG/KG-FITは有望だが、設定探索が必須**
   - テキストアンカー・階層制約で悪化を緩和できる可能性がある一方、
     未調整だと一般指標側の悪化も起こり得る。

---

## 7. 今後の課題と計画（次に何をするか）

### 7.1 KG-FIT(PairRE) の安定化と両立点探索

- 目的: 「ターゲット改善」だけでなく、MRR/Hitsの悪化を抑える設定を探す。
- 方向性:
  - 正則化強度（anchor/cohesion/separation）や neighbor_k のスイープ
  - incident triples（中間ノード周辺追加）の上限制御の影響切り分け

参照:
- REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001
- RULE-20260123-KGFIT_PAIRRE-001

### 7.2 Web取得（LLMKnowledgeRetriever）を反復精錬に統合

- 目的: ローカル候補集合（train_removed）に依存せず、外部情報源から「根拠トリプル」を取得して精錬できるようにする。
- 現状: 設計・実装計画を策定済み（local/web切替、provenance、キャッシュ、ID衝突回避など）。

参照:
- REC-20260123-ARM_WEB_RETRIEVAL-001

### 7.3 統計的比較の強化

- 現状のランダムベースラインは seed 反復（まず5、可能なら10）で分散評価する方針。
- UCB改善が「ランダムの何パーセンタイルに入るか」を提示し、主張を強固にする。

参照:
- REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001

---

## 8. 参考文書（台帳に基づく主要リンク）

本報告は docs/database/index.md の記載を基に、以下の文書を主要参照としてまとめた。

- v3概要（ルール駆動）: RULE-20260111-KG_REFINE_V3_OVERVIEW-001
- combo-bandit（arm運用）: RULE-20260111-COMBO_BANDIT_OVERVIEW-001
- relation priors（witness重み付け）: RULE-20260118-RELATION_PRIORS-001
- full pipeline（結果集約）: REC-20260118-FULL_PIPELINE_RESULTS-001
- add-allベースライン（悪化確認）: REC-20260118-ADD_ALL_TRAIN_REMOVED-001
- KG-FIT標準仕様（TAKG）: RULE-20260119-TAKG_KGFIT-001
- full pipelineのKG-FIT再評価: REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001
- drop_ratio vs add-all（再検証）: REC-20260120-DROP_RATIO_1_ADD_ALL-003 / REC-20260120-DROP_RATIO_1_ADD_ALL_KGFIT-003
- UCB priors=off（TransE vs KG-FIT(PairRE)）: REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001
- ランダム追加ベースライン（KG-FIT(PairRE)）: REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001
- Web取得統合計画: REC-20260123-ARM_WEB_RETRIEVAL-001
- KG-FIT(PairRE)運用標準: RULE-20260123-KGFIT_PAIRRE-001
