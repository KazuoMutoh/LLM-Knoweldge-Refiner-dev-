## 1. 緒言
知識グラフ（Knowledge Graph; KG）は、検索・推薦・推論・対話など多様な下流タスクの基盤となる。一方で、KGは多様な構造化・半構造化・非構造化データから（半）自動的に構築されることが多く、欠損（missing）・誤り（erroneous）・陳腐化（outdated）・矛盾（conflict）といった品質問題を内在しやすい。こうした品質問題は、KGを用いる下流タスクの性能と信頼性を直接損なう。

第一に、KGは閉じた系ではなく、内部構造だけでは埋まらない欠損が存在する。KGの欠損には、単に「推論が難しい」だけでなく、「元の情報源に存在するがKGに取り込まれていない」あるいは「そもそも取り込むべき情報源がKGの外にある」タイプが含まれる。このため、KG内部の情報のみに依存するリンク予測は、既存構造から整合する関係を推定できても、外部にしか根拠がない事実を十分な裏付けのないまま新規知識として確定することには限界がある（Paulheim 2017; Xue et al. 2022）。

第二に、リンク予測・異常検知の多くは、KG内部のトリプルと近傍構造・統計に強く依存するため、KGが疎であるほどモデルが参照できる証拠が乏しくなる。モデルの表現力を高めるだけでは、学習・推論の入力となる観測（トリプル）自体が不足している状況を根本的に解決できない。したがって、外部情報源から根拠（evidence）を収集し、トリプルとしてKGへ統合することは、学習・推論に供する観測を増やす直接的な方策である。

第三に、外部情報源は広大であり、どの追加が有益かはターゲット関係や既存KGの状態に依存する。そのため、追加を一度で完結させるよりも、根拠収集と追加を段階的（反復的）に行い、途中で得られる効果を踏まえて次の探索方針を調整する設計が自然である。

### 1.1 本研究の立場と貢献
本研究は、外部情報源から得られる根拠（evidence）を活用しつつ段階的にトリプルを追加し、欠損補完を通じて最終的なリンク予測性能（Hits@k（Hit率）、Mean Reciprocal Rank（MRR）等）およびターゲット関係の品質を改善する枠組みを提示する。既存の多くの枠組みがKG内部の観測に基づいて推定・補完を行うのに対し、本研究は「Knowledge Graph Embedding（KGE）に基づくリンク予測の品質を高めるために、KGの外部から根拠を取得して観測そのものを増やし、KGへ統合する」という立場をとる。

本研究の主な貢献は次の通りである。

- ターゲット関係に焦点を当て、外部情報源から証拠トリプルを取得しKGへ統合する反復的リファイン枠組みを定式化する。
- Hornルールに基づく証拠探索（acquire）を組み込み、候補ルール集合の選択を逐次意思決定（多腕バンディット）として扱うことで、限られた予算下で取得方針を適応的に更新する。
- Web由来候補の統合において、(i) Web内の安定ID化と (ii) 既存KGへのリンク（検索による類似候補取得とLarge Language Model（LLM）による同一性判定）を含む entity resolution を導入し、外部候補の一貫した統合を可能にする。
- 反復後のKGE再学習により、追加された観測がターゲット関係のリンク予測精度に与える影響を評価し、代理報酬として反映する実装指針を与える。

## 2. 関連研究

KGの品質改善に関する研究動向は、欠損補完（completion）と誤り検出・訂正（cleaning/refinement）の観点から整理されてきた（Paulheim 2017; Xue et al. 2022）。近年はこれに加えて、Large Language Model（LLM）の発展により、KG外の知識（Web・文書・複数ソース）を取り込みながらKGを拡張・補完する枠組みが急速に拡大している。特に、(i) Knowledge Graph Embedding（KGE）およびKnowledge Graph Completion（KGC）の改善に向けた外部知識の利用、(ii) Open-World前提での多ソース統合と信頼性推定、(iii) 反復的な取得・統合（監査可能な更新）という方向が顕著である。LLMを用いたKG構築・更新の研究動向はサーベイでも体系化されつつある。([Bian 2025][1])

本節では、既存研究を「(1) KGE/KGC（リンク予測）」「(2) 外部知識の取得とOpen-World統合（信頼性推定を含む）」「(3) HITL／アクティブラーニング」「(4) ルールベース」に整理し、その上で本研究の位置づけを明確化する。

### 2.1 KGEとリンク予測（KGCを含む）

知識グラフ埋め込み（KGE）は、エンティティ・関係を連続空間に写像し、スコア関数でトリプルの尤もらしさを評価することでリンク予測を行う。特に、**KGのグラフ構造（観測済みトリプル集合）のみ**から埋め込みを学習する代表的手法として、距離ベースのTransE（Bordes et al. 2013）、双線形スコア関数に基づくDistMult（[Yang et al. 2015][6]）、複素数埋め込みによるComplEx（[Trouillon et al. 2016][7]）、関係を複素空間での回転として表すRotatE（[Sun et al. 2019][8]）などが提案されてきた。一方で、**KGが疎な場合や、KG外に根拠が存在する事実に対して、観測済みトリプルと局所構造のみで推定する限界**が指摘されてきた（Paulheim 2017）。

この限界に対し近年は、テキスト情報やLLM由来知識をKGE/KGCへ統合する方向が進んでいる。代表的に、KG-FITは、構造のみのKGEを出発点としつつ、LLM支援の「エンティティクラスタの階層構造」を構成し、さらにエンティティのテキスト情報も併用してKGEをファインチューニングすることで、オープンワールド知識（グローバルな意味）とKG局所構造（ローカルな意味）の双方を埋め込みに反映する。([Jiang et al. 2024][2])

本稿では、エンティティにテキスト属性（名称・説明文など）が付与された知識グラフを **Text-Attributed Knowledge Graph（TAKG）** と呼び、TAKG上で埋め込みを学習する枠組み（テキスト属性を学習に統合するKGE）を **Text-Attributed Knowledge Graph Embedding（TAKGE）** と呼ぶ。KG-FITは、TAKGEを具体的に実現する手法の一つであり、テキスト由来のアンカーと階層構造に基づく正則化により、疎なKGや新規エンティティを含む状況での学習安定化を狙う。
また、オープンワールド仮定下の生成系KGCでは、LLMの幻覚による信頼性低下や、未知エンティティを含む新規事実の評価の難しさが課題となる。MusKGCは、関係テンプレートとエンティティ型制約により構造知識を自然言語へ接続し、KG内部の事実と外部の信頼できる知識を組み合わせて、根拠付きで欠損エンティティを生成する枠組みを提案している。さらに、新規事実のfactualityとconsistencyを検証する評価戦略も提示している。([Song et al. 2025][3])
さらに、DrKGCは、各クエリごとに「学習した論理規則」に誘導されたボトムアップのグラフ検索でサブグラフを動的に取得し、取得サブグラフをGCNアダプタで表現へ取り込んだ上で、その構造コンテキストをプロンプトへ統合してLLMのファインチューニングを行う枠組みを提案している。([Xiao et al. 2025][4])

これらの流れは、欠損補完器としての **KGE/KGCモデル**（スコア関数の表現学習）や、生成系KGCで用いる **LLM**（生成・推論過程）といった「補完モデル」を強化することで欠損補完を改善するアプローチである。一方で本研究は、KGEそのものを高度化するというより、**外部情報源から得た“確定的な周辺事実（evidence）”をKGへ統合して観測を増やし、その結果としてリンク予測の学習条件を改善する**立場を採る点で異なる。


### 2.2 外部知識の取得・統合とOpen-World補完（信頼性推定を含む）

KGの欠損や陳腐化に対し、KG外部（Web、文書、複数データソース）から知識を取得して統合する研究が拡大している。LLMを用いたKG構築・更新は、従来の「抽出→統合→融合」というパイプラインを、より言語主導・生成主導に再編する潮流として整理されている。([Bian 2025][1])

一方で、外部ソース由来知識はノイズや矛盾、古さを含み得るため、「何をどの程度信頼して統合するか」が中心課題となる。Open-WorldのKG completionでは、外部から得た主張（claims）に対して、負例を意識した表現学習や複数ソース信頼性推定（multi-source reliability inference）を組み合わせ、主張の品質を高める枠組みが提案されている。([Song et al. 2025][3])
また、LLMの幻覚や更新遅れを補う目的で、KGや外部知識を検索して生成を支えるRetrieval-Augmented Generation（RAG）に基づく枠組みも多数提案されている（主に下流タスクだが、取得・統合設計という観点で関連）。([Li et al. 2025][5])

本研究の web 方式は、反復ごとにWebから候補トリプルを収集し、entity resolution により既存KGへ整合的に結合する点で、上記の「外部知識統合」と問題意識を共有する。ただし本研究は、外部由来候補をただ追加するのではなく、**ルールに基づく evidence 探索（body 側の具体トリプル収集）を通じて、KGに追加する対象を“根拠トリプル”へ限定する**ことで、ノイズ注入と誤追加連鎖の抑制を狙う（head は保留）。


### 2.3 HITL／アクティブラーニングに基づくKGリファイン

Human-in-the-Loop（HITL）やアクティブラーニングでは、限られた予算の下で「どの候補を検証・追加・修正すべきか」を最適化し、効率的にKG品質を改善することを目指す（Xue et al. 2022）。近年は、KG更新を反復プロセスとして扱い、監査可能性（provenance・ログ）やワークフロー全体を重視する議論も増えている。例えば、Wikidataのようなコミュニティ主導のKGでは、各主張（statement）に出典（reference）を付与し、更新履歴を通じて変更過程を追跡できる設計が採用されている（Vrandečić and Krötzsch 2014）。研究動向としても、LLMを用いたKG構築・更新の枠組みが「取得→統合→検証・更新」を反復するプロセスとして整理されつつある。([Bian 2025][1])

ただし、多くのHITL/KG refinement研究は、候補生成や不確実性推定の段階でKG内部情報に強く依存しがちであり、KG外に根拠がある欠損に対しては「問い合わせ対象の選択」だけでは不足する場合がある。本研究は、HITLの代替として人手を直接使うのではなく、**外部情報源（Web等）を“根拠取得オラクル”として扱い**、その取得方針を逐次意思決定（多腕バンディット）で適応的に更新する点で、HITLにおける予算制約下の改善という考え方を外部根拠取得へ接続する。


### 2.4 ルールベース（マイニング）による補完・説明

ルールベース手法は、頻出パターンの抽出や論理規則の適用により、補完候補の生成や説明可能性に強みを持つ。AMIE系は、不完全な知識ベースから規則を抽出し、補完・検証に利用できる代表例である（Galárraga et al. 2013）。ルールは人間に解釈しやすく、候補提示の説明に有効である一方、ルール適用のみではKG外部の根拠を直接取り込めない。

近年は、ルール抽出側の品質向上や、埋め込み・外部情報と組み合わせたハイブリッド化も進んでいる（例：構造表現の強化を通じて推論・抽出を安定化する方向）。([Xiao et al. 2025][4])
本研究は、ルールを「head を生成して補完する」ためではなく、**外部候補集合上で body を満たす具体トリプル（evidence）を探索・収集するための“探索バイアス”として用いる**点に特徴がある。さらに、ルールの組（arm）を選択単位とし、予算下での探索を多腕バンディットとして扱うことで、外部根拠取得の効率化を図る。


### 2.5 本研究の位置づけ

以上を踏まえ、本研究は、(i) KG内部の推定に閉じず、(ii) ルールにより探索空間を制御しながら外部情報源から evidence を収集し、(iii) evidence のみを段階的に統合して観測を増やす、という更新方針に立つ。最終的な評価はターゲット関係のリンク予測性能（KGE/KGC）で行うが、更新後の再学習手順（例：KG-FITのようなテキスト・階層知識統合系バックエンドの適用タイミングと設定）は提案手法節で具体化する。([Jiang et al. 2024][2])
これは、LLMによる生成補完やRAGにおける回答生成支援とは異なり、KGそのものの学習条件（観測の疎性・局所構造）を外部根拠の統合で直接改善する方向から品質改善を捉え直す試みである。加えて、Open-World統合で重要となる信頼性推定・監査可能性の観点とも整合し、外部候補の provenance 保存や名寄せ（entity resolution）を含む運用可能な枠組みとして位置づけられる。([Bian 2025][1]; [Song et al. 2025][3])

## 3. 問題設定

本研究が扱うのは、不完全な知識グラフに対して外部情報源から追加トリプルを逐次的に獲得し、最終的なKGE（知識グラフ埋め込み）に基づくリンク予測精度を高める問題である。概念的には、(i) 現在のKGとモデル状態にもとづいて次に調査すべき外部情報（および追加すべき知識）を選択し、(ii) 外部から得られた知識をKGへ追加し、(iii) 更新後のKGでKGEを再学習してターゲット関係の予測精度を改善する、という意思決定を、限られた予算（問い合わせ回数や追加可能なトリプル数）のもとで反復する。

以下では、この逐次的なKG拡張を強化学習（あるいは逐次意思決定）問題として定式化する。

### 3.1 記号と評価
エンティティ集合を $\mathcal{E}$、関係集合を $\mathcal{R}$ とし、トリプルは $(h,r,t)\in\mathcal{E}\times\mathcal{R}\times\mathcal{E}$ で表す。初期の観測済みトリプル集合（初期KG）を $\mathcal{T}_0$ とする。本研究では、特に関心のあるターゲット関係 $r^*\in\mathcal{R}$ を1つ与え、その関係に関するリンク予測精度を改善する。

時刻 $t$ のKG $\mathcal{T}_t$ に対して、学習（または更新）によって得られるKGEモデルを $\theta_t$ と表し、学習手続きを $\mathrm{Train}$ として抽象化する。

$$
\theta_t\;=\;\mathrm{Train}(\mathcal{T}_t).
$$

ターゲット関係 $r^*$ に関して評価したいクエリ集合（例：評価用のヘッド／テール予測問題）を $\mathcal{D}_{\mathrm{target}}$ とし、Hits@k、MRRなどの評価指標をまとめた評価関数を $\mathcal{M}(\theta,\mathcal{D}_{\mathrm{target}})$ と書く。

### 3.2 逐次追加（強化学習的定式化）
エピソード長を $T$ とする。時刻 $t$ の状態 $s_t$ は、現在のKGと（必要なら）その時点のモデル状態を含むものとし、

$$
s_t\;=\;(\mathcal{T}_t,\theta_t)
$$

と表す（実際にはこの状態の一部を特徴量として用いる）。エージェントは、外部情報源にアクセスして新たな知識を得るための行動 $a_t$ を選択する。

$$
a_t\;\sim\;\pi(a\mid s_t).
$$

ここで $a_t$ は「どの対象について外部情報を調べるか」「どの形式で知識を獲得するか」といった外部情報獲得の選択を表す抽象的な行動であり、具体的な検索クエリの構成や検証手続きは手法節で述べる。

行動の結果として、外部情報源から追加されるトリプル集合 $\Delta_t$ が得られ、KGは

$$
\mathcal{T}_{t+1}\;=\;\mathcal{T}_t\cup\Delta_t
$$

と更新される。$\Delta_t$ は外部情報源の性質や選択した行動に依存して確率的に決まるとみなし、遷移分布 $P(\Delta_t\mid s_t,a_t)$ によって抽象化する。

更新後に再学習したモデル $\theta_{t+1}=\mathrm{Train}(\mathcal{T}_{t+1})$ を評価し、逐次意思決定の報酬として、例えば評価の増分

$$
r_t\;=\;\mathcal{M}(\theta_{t+1},\mathcal{D}_{\mathrm{target}})-\mathcal{M}(\theta_t,\mathcal{D}_{\mathrm{target}})
$$

を用いる。

ただし、$\theta_{t+1}$ の学習（特にTAKGEの再学習）は計算コストが大きく、各時刻で $r_t$ を厳密に計算することは現実的でない。したがって提案手法では、反復中は再学習を行わず、取得したevidenceの統計に基づく代理報酬で arm を評価・選択し、再学習は反復後（またはバッチ）にまとめて行う（4章）。

### 3.3 目的と制約（予算）
目的は、予算制約の下で方策 $\pi$ を選び、最終時刻の評価を最大化することである。

$$
\max_{\pi}\;\mathbb{E}\big[\mathcal{M}(\theta_T,\mathcal{D}_{\mathrm{target}})\big]
$$

$$
\text{s.t.}\quad \sum_{t=0}^{T-1} c(a_t,\Delta_t)\le B.
$$

ここで $c(a_t,\Delta_t)$ は外部情報獲得・検証・追加に要するコストを表し、問い合わせ回数制約や追加トリプル数制約などは $c$ と $B$ の選び方で表現できる（例：$c\equiv 1$ による問い合わせ回数、$c\equiv |\Delta_t|$ による追加数）。以降の節では、この問題設定の下で、外部情報から有益なトリプルを獲得して段階的にKGを拡張し、ターゲット関係のKGEに基づくリンク予測精度を改善する具体手法を述べる。

## 4. 提案手法

本研究の狙いは、ターゲット関係 $r^*$ のトリプルを外部情報から直接「確定」して追加するのではなく、まずは $r^*$ を説明し得る周辺事実（証拠トリプル; evidence）を段階的に集めて KG の観測を増やし、その後にまとめて KGE を再学習することで、最終的なリンク予測精度を改善する点にある。各反復では、どのルール群から探索を進めるべきかを逐次に決める必要があるため、3章で定式化した能動学習/強化学習（逐次意思決定）の枠組みに沿って、arm（ルールの組）を行動として選択しながら改善を進める。

以下では実装に準拠しつつ、4.1〜4.7の構成で、反復処理フロー、Hornルールによる証拠取得、TAKGE（KG-FIT）設定、armの構成と生成、arm選択（代理報酬・LLM-policy）、Web由来の候補拡張、反復後の再学習を述べる。

> **Running example（国籍予測）**：以降、ターゲット関係を $r^*=\texttt{/people/person/nationality}$ とし、人物 $p_0$ と国 $c_0$ からなるターゲットトリプル $(p_0,r^*,c_0)\in\mathcal{T}_{\mathrm{target}}$ を1つ固定して説明する。これは評価用に保持された（学習用KGからは除去された）真のトリプルであり、反復中は **head側 $(p_0,r^*,c_0)$ 自体はKGへ追加しない**。代わりに、$p_0$ の出生地や居住地など、国籍を支持し得る周辺事実（evidence）を収集・追加し、反復後に再学習したモデルが $c_0$ をより上位に順位付けできることを狙う。

### 4.1 反復処理フロー（手法の概要）
反復 $t=1,\dots,T$ の処理は、（観測）→（arm選択）→（証拠取得）→（代理評価）→（KG更新）を繰り返す逐次意思決定として記述できる。

1. 初期ルールプール $\mathcal{P}$ と armプール $\mathcal{A}$ を準備する（4.4）。
2. arm選択: 方策 $\pi$（UCB/ε-greedy/LLM-policy/Random）により $k$ 個のarmを選ぶ（4.5）。
3. 証拠取得（acquire）: 選択armについて、ターゲットトリプル集合から一部をサンプリングし、Hornルールのbodyを満たす証拠トリプル（evidence）を候補グラフ上で探索・収集する（4.2）。候補グラフはローカル候補集合または Web 取得候補（4.6）を合成して構成する。
4. 代理評価（evaluate）: witness（説明の厚み）と、新規に得られたevidence数等からarmごとの報酬を計算し、履歴を更新する（4.5）。
5. KG更新: 反復中に KG へ確定追加するのは evidence のみに限定し、ターゲット関係 $r^*$ の候補トリプルは保留集合（pending hypothesis）として保存する。

反復の成果（選択arm、evidence/追加トリプル、witness統計、方策テキスト等）は各 $t$ ごとに保存し、最終的に更新KGでKGEを再学習して評価する（4.7）。

> **Running example（反復1回の流れ）**：$t=1$ で、方策が arm $a_1$（国籍を支持し得るルール2本の組）を選んだとする。acquire により $(p_0,r^*,c_0)$ を「説明できる」body側のパスを候補グラフから探索し、例えば $(p_0,\texttt{/people/person/place\_of\_birth},z)$ や $(z,\texttt{/location/location/containedby},c_0)$ が新規に見つかれば evidence として追加する。反復中は $(p_0,r^*,c_0)$ は pending のまま保持し、代理報酬（witness・新規evidence数）で $a_1$ を評価して次の arm 選択へ進む。

### 4.2 Hornルールと証拠取得（acquire）
本研究で用いる Horn ルールは、知識グラフ上の関係パターンを「body（前件）を満たすなら head（結論）が成り立ちやすい」という形で表現する規則である。ターゲット関係 $r^*$ に対しては一般に
$$
h: B(x,y,\mathbf{z}) \Rightarrow (x,r^*,y)
$$
の形をとる。ここで $B(x,y,\mathbf{z})$ は、複数の原子式（トリプルパターン）の連言
$$
B(x,y,\mathbf{z}) = (x,r_1,z_1)\wedge(z_1,r_2,y)\wedge\cdots
$$
として書ける。

本手法の重要な点は、Hornルールを $(x,r^*,y)$ を直接追加するために使うのではなく、$(x,r^*,y)$ を支持し得る body 側の具体トリプル（evidence）を探索・取得するために用いる点にある。すなわち反復中にKGへ確定追加するのは body 由来の evidence のみであり、head 側の候補は保留仮説として記録する（4.5）。

取得（acquire）は、各arm（ルールの組）とサンプルされたターゲット $(x_0,r^*,y_0)$ に対し次を行う。

1. $x\leftarrow x_0,\;y\leftarrow y_0$ を固定する。
2. 候補グラフ上で $B(x_0,y_0,\mathbf{z})$ を満たす代入 $\mathbf{z}$ を探索する。
3. 代入が見つかるたび、そのbodyを構成する具体トリプル集合を evidence として収集する。

同一ターゲットに対して $\mathbf{z}$ の取り方が複数存在すれば、その数が witness（body を満たす置換数）であり、「その仮説がどれだけ多様な根拠パスで説明されるか」を表す。evidence のうち現 KG に存在しないものがあれば、それらが反復で追加される観測となる。

以下の説明では、例としてターゲット関係を国籍（$r^*=\texttt{/people/person/nationality}$）とし、$x$ を人物、$y$ を国として扱う。

> **Running example（ルールとevidence）**：例えば次のようなルールを想定する。
>
> $$
> h_1: (x,\texttt{/people/person/place\_of\_birth},z)\wedge(z,\texttt{/location/location/containedby},y)\Rightarrow (x,r^*,y)
> $$
>
> acquire では $(x,y)=(p_0,c_0)$ を固定し、候補グラフから $z$（出生地）を探索する。$z$ の候補が複数見つかれば witness が増え、各 $z$ に対応する body の具体トリプルが evidence として収集される（ただし $(p_0,r^*,c_0)$ は追加しない）。

### 4.3 TAKGE（KG-FIT）の必要性と学習目的
本研究では、外部情報源から得た周辺事実（evidence）を段階的に追加し、反復後にKGEを再学習してターゲット関係 $r^*$ のリンク予測性能を改善する。ここで重要なのは、「意味的に妥当なevidenceを追加すること」と「KGEの精度向上」が必ずしも一致しない点である。

構造のみのKGE（例: TransE）は、観測済みトリプル集合の近傍構造から埋め込みを学習するため、外部から集めたevidenceが“意味的には有益”であっても、(i) ターゲット関係の判別に必要な局所構造を十分に補わない、(ii) 追加トリプルがハブ関係やデータ固有の偏りを強め、却って誤った帰納を誘発する、(iii) 新規エンティティが少数のエッジしか持たない場合に表現が不安定になり、追加した事実が学習信号として効きにくい、といった理由で、精度向上に結びつかない場合がある。すなわち「意味的にKGを充実させる」ことと「構造のみKGEの最適化」が乖離し得る。

この乖離は、本研究の 4.2 のように「ターゲットトリプルを説明する典型的パターン」を Horn ルールとして用い、同型の evidence を大量に追加する場合に顕在化しやすい。4.2の取得は、ターゲット $(x_0,r^*,y_0)$ を固定して body を満たす $\mathbf{z}$ を探索し、bodyを構成する具体トリプルを evidence として収集する。ここで、body が同じ述語列（同じ構造パターン）に偏ると、追加されるトリプルは「構造としては似た制約」を大量に増やすことになり、構造のみKGEではエンティティを区別する手がかり（特徴差）を弱めてしまう。

TransEを例に、なぜ「妥協案的な埋め込み」へ寄りやすいかを数式で示す。TransEのスコアは
$$
f_{\theta}(h,r,t)=-\lVert \mathbf{e}_h+\mathbf{r}-\mathbf{e}_t\rVert
$$
であり、正例トリプル集合 $\mathcal{T}$ に対し（負例サンプリングを含むランキング損失の代わりに見通しのよい平方損失で近似すると）
$$
\min_{\{\mathbf{e}_e\},\{\mathbf{r}\}}\;\sum_{(h,r,t)\in\mathcal{T}}\lVert \mathbf{e}_h+\mathbf{r}-\mathbf{e}_t\rVert^2
$$
を解くことに対応する。

4.2の取得により、例えば共通の中間ノード $z$ を介した同型パターン
$$
(x_i,r_1,z),\qquad (z,r_2,y_i)\qquad (i=1,\dots,n)
$$
が大量に追加されるとする（同じ $r_1,r_2$ を共有し、$z$ は多くの $x_i$ と接続される「ハブ」になりやすい）。このとき第1項の和は
$$
\sum_{i=1}^n \lVert \mathbf{e}_{x_i}+\mathbf{r}_1-\mathbf{e}_{z}\rVert^2
$$
となり、$\mathbf{e}_z$ の最適解（他項を固定したときの最小二乗解）は
$$
\mathbf{e}_z\;\approx\;\frac{1}{n}\sum_{i=1}^n(\mathbf{e}_{x_i}+\mathbf{r}_1).
$$
すなわち、$z$ に多数のエッジが集まるほど、$\mathbf{e}_z$ は平均（妥協）に近づき、それに合わせて各 $\mathbf{e}_{x_i}$ も $\mathbf{e}_z-\mathbf{r}_1$ の近傍へ引き寄せられる。同様に、第2項
$$
\sum_{i=1}^n \lVert \mathbf{e}_{z}+\mathbf{r}_2-\mathbf{e}_{y_i}\rVert^2
$$
により、$\mathbf{e}_{y_i}$ も $\mathbf{e}_z+\mathbf{r}_2$ の近傍へ集約されやすい。

結果として、多数のエンティティ埋め込みが「同じパターン制約を満たすための平均的な位置」へ収束しやすく、エンティティ間の差分（判別性）が失われる。この状況では、正例と負例のスコア差が縮まりやすく、ランキング損失（マージン）を満たすことが難しくなるため、ターゲット関係 $r^*$ のリンク予測指標（Hits@k/MRR）が改善しない、場合によっては悪化する（= 予測が一様化し、誤りが増える）ことが起こり得る。

そこで本研究は、エンティティにテキスト属性を持つTAKGを前提に、TAKGE（KG-FIT）により、外部から得た周辺事実の“意味”を埋め込み学習に接続しやすくする。直観的には、テキスト由来の表現が新規エンティティや疎なエンティティの初期位置（アンカー）となるため、意味的に妥当なevidence追加が埋め込み空間の整合的な更新へ結びつきやすくなり、最終的に $r^*$ の予測改善と一致しやすい。

> **Running example（なぜTAKGEか）**：Web取得で新規の地名エンティティ $z$（例：小さな自治体）が現れ、$(p_0,\texttt{/people/person/place\_of\_birth},z)$ と $(z,\texttt{/location/location/containedby},c_0)$ が追加されたとする。さらに、同様のパターンが多数の人物に対して繰り返し追加されると、例えば
> $$
> (p_1,\texttt{/people/person/place\_of\_birth},z),\; (z,\texttt{/location/location/containedby},c_0)\\
> (p_2,\texttt{/people/person/place\_of\_birth},z),\; (z,\texttt{/location/location/containedby},c_0)\\
> \cdots
> $$
> のように、共通の $z$ を介した「同じ構造制約」$(p_i, r_1, z)$ と $(z, r_2, c_0)$ が大量に増える（$z$ がハブ化する）。このとき構造のみKGEでは、$z$ の表現が多くの制約の“平均（妥協）”に引き寄せられ、人物 $p_i$ も $z$ 近傍に集約されやすく、結果として $r^*$ の判別に効く差分が弱まる可能性がある。一方TAKGEでは、$z$ の名称・説明文から得るアンカー $\mathbf{a}_z$ と正則化により、疎な新規ノードでも意味的に妥当な位置へ配置されやすく、追加evidenceが $r^*$ の予測に寄与しやすい。

以下では、KG-FITに整合する形でTAKGEの学習目的を具体化する。構造のみKGEでは、エンティティ埋め込み $\mathbf{e}\in\mathbb{R}^d$ と関係埋め込み $\mathbf{r}$ を用い、スコア関数 $f_\theta(h,r,t)$（例: TransEなら $-\lVert \mathbf{e}_h+\mathbf{r}-\mathbf{e}_t\rVert$）と損失
$$
\mathcal{L}_{\mathrm{KGE}}(\theta)=\sum_{(h,r,t)\in\mathcal{T}}\ell\big(f_\theta(h,r,t),\;\mathcal{N}(h,r,t)\big)
$$
を最小化する（$\mathcal{N}$ は負例サンプリング、$\ell$ はランキング損失等）。

TAKGEではこれに加え、各エンティティ $e$ のテキスト（名称・説明文）から得たテキスト埋め込み $\mathbf{a}_e\in\mathbb{R}^d$ を用意し、エンティティ埋め込み $\mathbf{e}_e$ がテキスト由来の意味と整合するように正則化する。例えば、アンカー整合（anchor）として
$$
\mathcal{L}_{\mathrm{anchor}}=\sum_{e\in\mathcal{E}}\lVert \hat{\mathbf{e}}_e-\hat{\mathbf{a}}_e\rVert^2
$$
（$\hat\cdot$ は正規化）を加える。

さらにKG-FITでは、テキスト埋め込み上でのクラスタリングにより seed階層（クラスタ）を構成し、クラスタ中心 $\mathbf{c}_g$ と割当 $g(e)$ を用いて、同一クラスタ内での凝集（cohesion）
$$
\mathcal{L}_{\mathrm{cohesion}}=\sum_{e\in\mathcal{E}}\lVert \hat{\mathbf{e}}_e-\hat{\mathbf{c}}_{g(e)}\rVert^2
$$
と、近傍クラスタ集合 $\mathcal{N}(g)$ に対する分離（separation）
$$
\mathcal{L}_{\mathrm{sep}}=\sum_{g}\sum_{g'\in\mathcal{N}(g)}\max\big(0,\;\cos(\hat{\mathbf{c}}_g,\hat{\mathbf{c}}_{g'})-\tau\big)
$$
を導入し、意味的に近いものは近く、区別すべきものは過度に近づかないように制約する（$\tau$ は許容類似度）。以上より、TAKGE（KG-FIT）の目的は
$$
\mathcal{L}_{\mathrm{TAKGE}}=\mathcal{L}_{\mathrm{KGE}}+\lambda_a\mathcal{L}_{\mathrm{anchor}}+\lambda_c\mathcal{L}_{\mathrm{cohesion}}+\lambda_s\mathcal{L}_{\mathrm{sep}}
$$
の最小化として表せる。テキストアンカーと階層正則化により、外部から得たevidence追加（特に新規/疎なエンティティ周辺の事実追加）が埋め込み学習へ反映されやすくなり、構造のみKGEに比べて「意味的充実」と「精度向上」が一致しやすいことを期待する。

### 4.4 初期ルールプールとarmの生成
初期ルールプール $\mathcal{P}$ は、ターゲット関係 $r^*$ を結論部に持つ Horn ルール
$$
h: B(x,y,\mathbf{z}) \Rightarrow (x,r^*,y)
$$
を対象に、AMIE+（Galárraga et al. 2013）等の規則マイニングにより抽出し、support、head coverage、PCA confidence等の品質指標に基づいて上位集合として構成する。以降の反復では $\mathcal{P}$ は固定し、探索の単位をルール単体ではなく「ルールの組（arm）」とする。

armは複数ルールの組であり、同時適用する集合arm $a=\{h_{i_1},\dots,h_{i_m}\}$ を基本とする（順序適用による拡張も可能）。armを用いる利点は、(i) 同一仮説を異なる根拠パターンで厚く支持する、(ii) 片方のルールでは拾えない evidence をもう一方が補う、といった相補性を探索単位に取り込める点である。

初期arm生成では、まず各ルール $h$ を単独arm（singleton-arm）として採用する。次にルールペア $(h_i,h_j)$ を候補とし、ターゲットトリプル集合 $\mathcal{T}_{\mathrm{target}}$ と候補集合 $\mathcal{S}_{\mathrm{cand}}$（例：ローカルKGから構成した候補集合）に対して、各ルールが支持し得るターゲット集合 $\mathcal{S}(h)\subseteq \mathcal{T}_{\mathrm{target}}$ を「候補集合上でbodyが充足可能」という条件で定める。共起スコアとして
$$
\mathrm{cooc}(h_i,h_j)=\frac{|\mathcal{S}(h_i)\cap \mathcal{S}(h_j)|}{|\mathcal{S}(h_i)\cup \mathcal{S}(h_j)|}
$$
（Jaccard係数）を計算し、上位のペアを pair-arm として採用する。これにより、同一結論に対する複数根拠（witness構造）を得やすいarmを初期探索候補に含める。

> **Running example（armの例）**：$\mathcal{P}$ に $h_1$（出生地→国）と、別経路の $h_2$（居住地→国 など）が含まれているとする。$h_1$ と $h_2$ が同じターゲット集合 $\mathcal{S}(h)$ を多く共有する場合、それらを組にした pair-arm $a=\{h_1,h_2\}$ は $(p_0,r^*,c_0)$ を複数の経路で説明でき、witness を稼ぎやすい候補として優先的に探索される。

### 4.5 arm選択

#### 4.5.1 代理報酬
本来、逐次意思決定の報酬は「その反復で追加したトリプル集合が、再学習後の評価 $\mathcal{M}(\theta,\mathcal{D}_{\mathrm{target}})$ をどれだけ改善したか」で与えられる。しかしTAKGE（KG-FIT）の再学習は計算コストが大きく、各反復で再学習して報酬を観測することは実運用上難しい。そこで提案手法のアイデアとして、反復中はKGE/TAKGEの再学習を行わず、arm の良さは代理報酬（proxy reward）で評価する。

本実装では、arm $a$ の報酬を witness の合計と、新規に追加できた evidence 数の線形結合として
$$
R(a)=\lambda_w\sum_{t\in\mathcal{T}_{\mathrm{sample}}}\mathrm{witness}(a,t) + \lambda_e\,|\mathrm{NewEvidence}(a)|
$$
で定義する。witness は「複数の grounding により説明される厚み」を表し、偶然一致に依存しにくい。一方で NewEvidence は、現KGが欠いていた周辺事実をどれだけ補完できたかを直接数える指標であり、「headを保留し body 由来の観測を増やす」という更新方針と整合する。

ただし witness はハブ的関係により水増しされ得るため、関係ごとの prior $X_r$ による重み付けを導入できる。ルール $h$ の重みを body predicate の prior の積
$$
W(h)=\prod_{p\in\mathrm{body\_predicates}(h)} X_p
$$
で定義し、witness を $\sum_{h\in a} W(h)c_h(t)$ の形で集計する。

> **Running example（報酬計算）**：arm $a=\{h_1,h_2\}$ を適用し、サンプルしたターゲット $(p_0,r^*,c_0)$ に対して body の grounding を探索したとする。例えば $h_1$ が
> $$
> h_1:(x,\texttt{/people/person/place\_of\_birth},z)\wedge(z,\texttt{/location/location/containedby},y)\Rightarrow (x,r^*,y)
> $$
> であり、候補グラフ上で $z$ が2通り見つかったとする（grounding数=2）：
> $$
> z=z_1:\; (p_0,\texttt{/people/person/place\_of\_birth},z_1),\; (z_1,\texttt{/location/location/containedby},c_0)\\
> z=z_2:\; (p_0,\texttt{/people/person/place\_of\_birth},z_2),\; (z_2,\texttt{/location/location/containedby},c_0)
> $$
> このとき witness は「bodyを満たす置換の個数」なので $\mathrm{witness}(a,(p_0,r^*,c_0))=2$ となる（arm内で複数ルールを使う場合は、各ルールの grounding 数の合計/和集合で集計される）。
>
> 次に NewEvidence は、「収集した body トリプルのうち現KGに存在しなかったもの」の本数で数える。例えば既にKGに $(p_0,\texttt{/people/person/place\_of\_birth},z_1)$ はあり、$(z_1,\texttt{/location/location/containedby},c_0)$ と $(p_0,\texttt{/people/person/place\_of\_birth},z_2)$ が未観測だったなら、反復で確定追加される新規evidenceは2本で $|\mathrm{NewEvidence}(a)|=2$ となる。
>
> したがって、このターゲットに対する代理報酬は $R(a)=\lambda_w\cdot 2+\lambda_e\cdot 2$（複数ターゲットをサンプルした場合はそれらの合計）となる。

#### 4.5.2 Arm選択
arm選択は、4.5.1で定義した代理報酬 $R(a)$ の履歴を用いて、探索（未知armの試行）と活用（有望armの反復）を両立させる問題として扱う。実装では、UCB、ε-greedy、LLM-policy（および比較用のRandom）を選択肢として用意する。

##### 4.5.2.1 UCB
UCBでは、arm $a$ の推定平均報酬 $\hat\mu(a)$ と試行回数 $n(a)$ に基づき
$$
\mathrm{UCB}(a)=\hat\mu(a)+\alpha\sqrt{\frac{\log t}{n(a)}}
$$
が最大となるarmを優先して選択する（Auer et al. 2002）。第2項は未試行・試行回数の少ないarmほど大きくなり、探索を促進する。

> **Running example（UCB）**：$a_1=\{h_1,h_2\}$ が高い $\hat\mu(a_1)$ を持つ一方、未試行の $a_2$ がある場合、UCBは $a_2$ の不確実性項を加味して一定割合で試し、$(p_0,r^*,c_0)$ を別の根拠パターンで説明できる可能性を探索する。

##### 4.5.2.2 ε-greedy
ε-greedyでは、確率 $\varepsilon$ で探索としてランダムにarmを選び、確率 $1-\varepsilon$ で活用として $\hat\mu(a)$ が最大のarm（または上位 $k$ 個）を選ぶ。これにより、探索の頻度を $\varepsilon$ で明示的に制御しつつ、平均報酬の高いarmへ試行を集中させる。

> **Running example（ε-greedy）**：ほとんどの反復で $a_1$ を選びつつ（活用）、一定確率で $a_3$（例えば「所属組織→所在地→国」）を試すことで、$p_0$ 周辺の未観測evidenceを取り逃がしにくくする。

##### 4.5.2.3 LLM-policy
LLM-policy は、平均報酬などの統計だけではなく、(i) ターゲット関係 $r^*$ の文脈（relation説明）、(ii) arm内ルールの body/head predicate とその自然言語説明、(iii) 直近反復の diagnostics（coverage/witness/overlap等）、(iv) 取得・追加されたトリプル例（target/evidence/added）を入力として、意味的整合性（semantic grounding）を明示的に評価しつつ次のarmを選ぶ。さらに、次回以降に再利用できる選択基準として policy_text を更新し、探索と活用のバランス（例: 高報酬だが意味が薄いarmの抑制、意味整合の高いarmへの重点化）を方針として記録する。

このとき、プロンプトには entity2text/relation2text により可読化したトリプル例を含め、LLMは（少なくとも）選択armのID列と更新後の policy_text を構造化出力として返す。加えて、armごとに「意味整合」「証拠関連度」「proxy指標の信頼性」といった観点の簡易ルーブリックで評価した上で選択理由を生成させ、統計だけに偏る選択を抑制する。

重要なのは、LLM-policy は arm の「選択」を担うのみであり、evidence の実取得は次節以降で述べる acquire により実行される点である。

> **Running example（LLM-policy）**：統計上は $a_4$ の $\hat\mu$ が高くても、取得されたevidenceが国籍と無関係（例：人物の趣味・作品など）なら、LLM-policy は「国籍を支持する関係（出生地/居住地/所属組織所在地など）を優先する」という policy_text を更新し、$(p_0,r^*,c_0)$ の説明に近い arm（例：$a_1$）を選びやすくする。

### 4.6 Webに基づく候補拡張
候補集合をローカルな知識グラフ内に限定すると、KGに未登場のエンティティ（新規地名・新規組織など）や、ローカルに存在しない周辺関係を介する根拠パスを取り込めない。この制約を緩和するため、本研究では各反復で Web 検索を併用し、LLM によって候補トリプル集合を外部から収集して候補グラフを拡張する。

反復 $t$ における Web 由来候補の取得は概ね次の手順で行う。

1. **予算制約**：問い合わせ回数や保持する候補トリプル数に上限を設け、コストとノイズを制御する。
2. **候補抽出**：ターゲットトリプルと Horn ルールの body パターン（または周辺関係の候補リスト）を入力として、body を満たし得る中間エンティティと、その成立に必要な候補トリプル群を Web から抽出する。各候補には出典（URL 等）を付与し、根拠を追跡可能にする。
3. **表記ゆれ・同一性の正規化**：Web 由来のエンティティは表記が揺れやすいため、(i) 表層形と出典情報に基づく決定論的な識別子付与（例：ハッシュ化）により同一候補を統合し、(ii) 既存KGのエンティティと同一である可能性が高い場合は entity linking により既存IDへ揃える。
4. **漏洩防止**：ターゲット関係 $r^*$（仮説）そのものを主張する候補は除外し、証拠探索は body 側の周辺関係の補完に限定する。
5. **保存と再現性**：取得した候補トリプルと出典情報（必要に応じて重複除去後）を保存し、後から取得根拠を監査できるようにする。

以降の evidence 探索では、候補グラフを「現KGと Web 由来候補集合の和集合」として構築し、4.2と同じ body マッチングにより evidence と witness を計算する。

> **Running example（Web候補）**：ローカルKGに $(z,\texttt{/location/location/containedby},c_0)$ が無い場合、Web から「$z$ は $c_0$ の都市である」と読める根拠を含む候補を収集し、表記ゆれを正規化した上で候補集合へ加える。ここでも漏洩防止のため、$(p_0,r^*,c_0)$（国籍そのもの）を直接述べる候補は除外し、body側の関係のみを探索空間に組み込む。

### 4.7 反復後の再学習
反復で追加された evidence（必要に応じて新規エンティティ周辺の incident triples を含む）を初期データセットへ反映し、更新KGでKGEを再学習して最終評価を行う。反復中は head（$r^*$）を保留するため、再学習では「周辺事実の増加」がターゲット関係の判別に寄与するかを評価することになる。再学習は TAKGE（4.3）の設定（KG-FITバックエンド）で実施し、最終的な Hits@k や MRR を算出する。

> **Running example（最終評価）**：反復により $p_0$ 周辺の出生地・所在地などの evidence が増えると、再学習後のモデルは $(p_0,r^*,y)$ の候補国 $y$ を再順位付けし、真の $c_0$ をより上位に位置づけられることが期待される（反復中は $c_0$ を直接追加していないため、改善は周辺事実の統合効果として解釈できる）。

## 5. 参考文献

[1] Haonan Bian. *LLM-empowered knowledge graph construction: A survey*. 2025.

[2] Pengcheng Jiang, Lang Cao, Cao Xiao, Parminder Bhatia, Jimeng Sun, and Jiawei Han. *KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge*. 2024.

[3] Xin Song, Liu Haiyan, Haiyang Wang, Ye Wang, Kai Chen, and Bin Zhou. *MusKGC: A Flexible Multi-source Knowledge Enhancement Framework for Open-World Knowledge Graph Completion*. 2025.

[4] Yongkang Xiao, Sinian Zhang, Yi Dai, Huixue Zhou, Jue Hou, Jie Ding, and Rui Zhang. *DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains*. 2025.

[5] Mufei Li, Siqi Miao, and Pan Li. *Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation*. 2025.

[6] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. *Embedding Entities and Relations for Learning and Inference in Knowledge Bases*. 2015.

[7] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. *Complex Embeddings for Simple Link Prediction*. 2016.

[8] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space*. 2019.

[9] Heiko Paulheim. *Knowledge Graph Refinement: A Survey of Approaches and Evaluation Methods*. 2017.

[10] Bingcong Xue and Lei Zou. *Knowledge Graph Quality Management: a Comprehensive Survey*. 2022.

[11] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. *Translating Embeddings for Modeling Multi-relational Data*. 2013.

[12] Denny Vrandečić and Markus Krötzsch. *Wikidata: A Free Collaborative Knowledge Base*. 2014.

[13] Luis Galárraga, Christina Teflioudi, Katja Hose, and Fabian Suchanek. *AMIE: Association Rule Mining under Incomplete Evidence in Ontological Knowledge Bases*. 2013.

[14] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. *Finite-time Analysis of the Multiarmed Bandit Problem*. 2002.

[1]: https://arxiv.org/abs/2510.20345 "LLM-empowered knowledge graph construction: A survey"
[2]: https://arxiv.org/abs/2405.16412 "KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge"
[3]: https://aclanthology.org/2025.emnlp-main.508/ "MusKGC: A Flexible Multi-source Knowledge Enhancement Framework for Open-World Knowledge Graph Completion"
[4]: https://arxiv.org/abs/2506.00708 "DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains"
[5]: https://openreview.net/forum?id=JvkuZZ04O7 "Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation"
[6]: https://arxiv.org/abs/1412.6575 "Embedding Entities and Relations for Learning and Inference in Knowledge Bases"
[7]: https://arxiv.org/abs/1606.06357 "Complex Embeddings for Simple Link Prediction"
[8]: https://arxiv.org/abs/1902.10197 "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"

[9]: https://doi.org/10.3233/SW-160218 "Knowledge graph refinement: A survey of approaches and evaluation methods"
[10]: https://doi.org/10.1109/TKDE.2022.3150080 "Knowledge Graph Quality Management: a Comprehensive Survey"
[11]: https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html "Translating Embeddings for Modeling Multi-relational Data"
[13]: https://doi.org/10.1145/2488388.2488425 "AMIE: Association Rule Mining under Incomplete Evidence in Ontological Knowledge Bases"
[14]: https://doi.org/10.1023/A:1013689704352 "Finite-time Analysis of the Multiarmed Bandit Problem"

[12]: https://doi.org/10.1145/2629489 "Wikidata"

