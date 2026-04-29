# RULE-20260111-COMBO_BANDIT_OVERVIEW-001: ルール駆動KGリファイン (v3-combo-bandit) 概要

作成日: 2026-01-11
最終更新日: 2026-01-28
移管元: [docs/algorithm_overview_v3_combo_bandit.md](../algorithm_overview_v3_combo_bandit.md)
参照: [RULE-20260111-KG_REFINE_V3_OVERVIEW-001](RULE-20260111-KG_REFINE_V3_OVERVIEW-001.md)

更新日: 2026-01-14
- `retrain_and_evaluate_after_arm_run.py` を更新: afterモデルを updated_triples で再学習するモード（`--after_mode retrain`）を追加し、従来の評価のみ（`--after_mode load`）も維持。

更新日: 2026-01-17
- LLM-policyのarm選択で、armの「意味」（body/head predicateの説明）と「説明度」（witness/coverage）・取得evidenceのサンプルを提示し、target relation文脈に整合するarmを選びやすくする運用を追記。

更新日: 2026-01-18
- witness の水増し（ハブ関係・KGE非フレンドな関係）を抑制するため、relation priors（$X_r$）で witness 寄与を重み付けする運用を追記。

更新日: 2026-01-23
- 実験知見（追加トリプルにより構造のみKGEで target score/Hits が悪化し得る）を踏まえ、TAKG（テキスト属性）前提のKGEとして KG-FIT を採用する位置づけを追記。
- KG-FITのinteraction modelとして TransE / PairRE を採用した点と、関連rules（KG-FIT/PairRE/priors/統合ランナー）へのリンクを追記。

更新日: 2026-01-28
- UCB vs Random(seed=0) の rerun1（3関係）で、head_coverage（文脈が付いたか）だけでは改善を説明できず、追加トリプルの predicate 種類が改善/悪化に偏り得る点を追記（記録: [docs/records/REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md](../records/REC-20260128-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md)）。

---

本ドキュメントは [RULE-20260111-KG_REFINE_V3_OVERVIEW-001](RULE-20260111-KG_REFINE_V3_OVERVIEW-001.md) の設計を踏襲しつつ、
- **埋込スコア/ベクトルを各反復で計算しない**
- **ルール単体ではなく「複数ルールの組み合わせ」を腕（arm）として選択する**
- 候補トリプルの受理・報酬を **(4) 機能的矛盾（衝突）** と **(5) 支持構造（witness）** に基づく代理指標で行う
という方針を明示した差分版（v3派生）である。

## 0. 実装対応状況（2026-01-23時点）
本ドキュメントの方針に対し、現在リポジトリには次が実装されている。

- arm表現と決定的ID:
  - [simple_active_refine/arm.py](../../simple_active_refine/arm.py)
  - `Arm.key()` は決定的キーを返し、`ArmWithId.create()` は `arm_id = "arm_" + sha1(Arm.key())[:12]` の形式で安定IDを生成する。
- arm履歴:
  - [simple_active_refine/arm_history.py](../../simple_active_refine/arm_history.py)
  - `ArmHistory` は arm 単位の報酬・追加トリプル等を記録し、統計を返す（pickle/JSON保存を含む）。
- arm選択（UCB/ε-greedy/LLM-policy/Random）:
  - [simple_active_refine/arm_selector.py](../../simple_active_refine/arm_selector.py)
  - LLM-policy はテストではLLM呼び出しをモックして外部APIに依存しない。
- 出力ディレクトリ規約:
  - 反復出力は `base_output_path/iter_k/` を標準とし、補助関数 `get_iteration_dir(base_output_path, k)` を [simple_active_refine/io_utils.py](../../simple_active_refine/io_utils.py) に用意している。

- arm-run後のKGE評価（学習済モデル使用）:
  - [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)
  - `accepted_evidence_triples.tsv` を集約して updated dataset を作成し、Hits@k/MRR と target score を比較する。
  - before: 学習済モデルを読み込む。
  - after: `--after_mode load` の場合は学習済モデルを読み込み、`--after_mode retrain` の場合は updated dataset で再学習して保存する。

- TAKG（KG-FIT）KGEバックエンド（TransE / PairRE）:
  - [simple_active_refine/embedding.py](../../simple_active_refine/embedding.py)（`embedding_backend="kgfit"`）
  - 標準仕様: [docs/rules/RULE-20260119-TAKG_KGFIT-001.md](RULE-20260119-TAKG_KGFIT-001.md)
  - PairRE運用標準: [docs/rules/RULE-20260123-KGFIT_PAIRRE-001.md](RULE-20260123-KGFIT_PAIRRE-001.md)

### 0.1 TAKG（テキスト属性）と KG-FIT 採用の位置づけ（実験知見）

本パイプラインは反復中にKGEの再学習/再スコアリングを行わないため、「追加されたトリプルがKGEにとってノイズになり、最終的な再学習で target score / Hits が悪化する」ケースを避ける必要がある。

観測された現象（要点）:
- `train_removed` 由来のトリプルを大量に追加すると、構造のみKGE（例: TransE）では **target score/Hits@k が悪化し得る**。
  - 例: 全投入（add-all）や full pipeline の after 再学習で悪化が観測（記録）: [docs/records/REC-20260118-ADD_ALL_TRAIN_REMOVED-001.md](../records/REC-20260118-ADD_ALL_TRAIN_REMOVED-001.md), [docs/records/REC-20260118-FULL_PIPELINE_RESULTS-001.md](../records/REC-20260118-FULL_PIPELINE_RESULTS-001.md)

このため本プロジェクトでは、TAKG（テキスト属性付きKG）を前提として、KGE側にテキストアンカー/階層正則化を統合する **KG-FIT** を採用する。

- KG-FITでは、固定テキスト埋め込みと seed 階層（クラスタ）由来の正則化（anchor/cohesion/separation）を損失へ加算し、追加トリプルによる局所的な過制約やスコア悪化を緩和することを期待する。
  - TransE（構造のみ）結果をKG-FITへ置換して再評価した記録: [docs/records/REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001.md](../records/REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001.md)

採用した interaction model（KG-FITバックエンド）:
- TransE（baseline）
- PairRE（relation表現を2本持つ）
  - priors=off を強制して TransE vs KG-FIT(PairRE) を比較する計画/結果: [docs/records/REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001.md](../records/REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001.md)

関連 rules:
- KG-FITバックエンド標準（TAKG）: [docs/rules/RULE-20260119-TAKG_KGFIT-001.md](RULE-20260119-TAKG_KGFIT-001.md)
- KG-FIT（PairRE）運用標準: [docs/rules/RULE-20260123-KGFIT_PAIRRE-001.md](RULE-20260123-KGFIT_PAIRRE-001.md)
- KG-FIT正則化/neighbor_k運用（速度・再現性）: [docs/rules/RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001.md](RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001.md)
- relation priors（$X_r$）: [docs/rules/RULE-20260118-RELATION_PRIORS-001.md](RULE-20260118-RELATION_PRIORS-001.md)
- 統合ランナー（Step0 priors含む）: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- arm pipeline 実装仕様（witness重み付け含む）: [docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md](RULE-20260117-ARM_PIPELINE_IMPL-001.md)

## 1. アルゴリズムの概要
- 目的: 既存KGに外部情報を統合しつつ、ターゲット関係 $r_*$（機能的・1対1寄り）に関する追加トリプルを段階的に採用し、**（計算資源の許すタイミングで）再学習**によりリンク予測性能（MRR等）を向上させる。
- 基本方針: 反復内では埋込モデルの再学習やスコア計算を行わず、候補トリプルの品質を **witness（説明の厚み）** と **矛盾（衝突）** を中心とする代理指標で評価する。
- 重要な変更点: ルール選択の単位を「ルール $h$」から「ルール組み合わせ $a$（arm）」に変更する。arm は body=2 ルールの集合または順序列として表現される。

## 2. アルゴリズムの詳細
記法: KGを $G_k=(E,R,T_k)$、ターゲット関係を $r_*$、Hornルールを $h: B(x,y,\mathbf{z}) \Rightarrow (x,r_*,y)$ とする。反復回数を $N$ とし、$k$-回目のKGを $G_k$ と表記する。

- ルールプールを $\mathcal{P}=\{h_1,\dots,h_M\}$ とする（主に body=2 のルール）。
- arm（取得ポリシー）集合を $\mathcal{A}=\{a_1,\dots,a_L\}$ とする。
- $k$ 回目に選ぶ arm の集合を $\mathcal{S}_k \subseteq \mathcal{A}$ とする。
- arm $a$ の適用で得られる候補集合を $C_a$、受理集合を $A_k$ とする。

### 2.1 初期ルールプール構築（v3と同様）
初期プール構築は [RULE-20260111-KG_REFINE_V3_OVERVIEW-001](RULE-20260111-KG_REFINE_V3_OVERVIEW-001.md) の 2.1 と同様に、AMIE+ 等で Horn ルール集合を抽出し、品質指標でフィルタしたものを $\mathcal{P}$ とする。

本派生では、**プールを原則 body=2 に制限**し、長いルール（body=3以上）は採用しない（または強いフィルタを課して限定的に扱う）。

### 2.2 初期arm（取得ポリシー）生成（banditループ前）
arm は「body=2 ルールの組み合わせ」を表す。表現は2通りを許す。

- **集合arm（同時適用）**: $a=\{h_{i_1},\dots,h_{i_m}\}$
- **順序付きarm（逐次適用）**: $a=(h_{i_1} \rightarrow h_{i_2} \rightarrow \cdots \rightarrow h_{i_m})$

初期arm集合 $\mathcal{A}$ は $\mathcal{P}$（body=2 ルール集合）を材料として、トリプル追加前のKG（例: $G_0$）の解析により構築する。

本派生では、反復精錬中に **ルール（$\mathcal{P}$）そのものを更新しない**（プールは固定）ことを原則とし、学習対象は arm（組み合わせ）である。

初期armは次のいずれか（または併用）で構築する。

1. **単純構築**: まず $m=1$（単独arm）を作り、上位Kルールから $m=2$（ペアarm）を作る。
  - 例: $\mathcal{A}\leftarrow\{\{h_1\},\dots,\{h_M\}\}\cup\{\{h_i,h_j\}\mid h_i,h_j\in\mathrm{TopK}, i<j\}$
  - TopK の基準は、初期段階では AMIE品質（support/head coverage/PCA confidence等）でよい。

2. **静的共起（witness/head候補ベース）構築**: トリプル追加前のKG（例: $G_0$）を解析して共起行列を作り、共起の強いペア/小集合を初期arm候補として採用する。
  - ここでの共起は「反復ログ」ではなく、KG上の成立状況を数える **静的（static）** な共起である。
  - 可能ならターゲット条件付け（例: $r_*$ を持つ head 集合、または高信頼近傍の head 集合）を行い、ターゲット周辺での共起に寄せる。
  - 共起の集計は witness に基づいてもよいし、head候補集合 $C_h$ の重なり（下記）に基づいてもよい。

#### 2.2.1 初期arm生成で用いるデータ（実装準拠）
本実装（[simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py), [build_initial_arms.py](../../build_initial_arms.py)）では、初期armの構築を「ターゲット集合に対する head-support の重なり」として定義する。

- **ターゲット集合** $\mathcal{T}_{\text{target}}$:
  - 形式: 3項組 $(x, r_*, y)$ の列（TSV）
  - 目的: 「どの head/triple を説明対象とみなすか」の固定集合
  - 実体: `target_triples.txt`（実験ディレクトリ配下など）
- **候補トリプル集合** $\mathcal{S}_{\text{cand}}$:
  - 形式: 3項組 $(s, p, o)$ の列
  - 目的: ルールbodyの充足判定（conjunctive query の満足可能性判定）にのみ使用
  - 実体: 通常は `train.txt` に `train_removed.txt` を任意で加えたもの（`--include-train-removed`）

この設計により、初期arm生成は「外部取得」「反復ログ」に依存せず、$G_0$ 由来の静的情報のみで再現可能となる。

#### witnessベース共起（初期arm候補生成のための集計）
候補 $(x,r_*,y)$ が生成される（またはKG上で成立する）際、ルール $h$ の body を満たす置換（substitution）の数を witness と呼ぶ。

- $c_h(x)$: head $x$ に対してルール $h$ が成立した witness 数
- $c_h(x,y)$: 候補ペア $(x,y)$ に対してルール $h$ が成立した witness 数

共起スコアの例（xベース・静的/動的どちらにも適用可能）:
$$
\mathrm{cooc}(h_i,h_j)=\frac{|\{x\mid c_{h_i}(x)>0\wedge c_{h_j}(x)>0\}|}{|\{x\mid c_{h_i}(x)>0\}|}
$$

共起の強い組を優先して arm 候補を生成する。

#### head候補ベース共起（初期arm候補生成のための集計）
witnessベース共起が「body成立（置換）の重なり」に基づくのに対し、head候補ベース共起は「生成された head 候補トリプルの重なり」に基づく。

各ルール $h$ に対して、ターゲット関係 $r_*$ の候補（head候補）集合を
$$
C_h=\{(x,r_*,y)\mid h\text{ により候補として生成された}\}
$$
とする（集合armの場合は各ルールの $C_h$、順序付きarmの場合は適用段階ごとに $C_h$ を得てもよい）。

このとき、ルール間の共起は次のような集合の重なりで評価できる。

- 条件付き共起（方向あり）:
$$
\mathrm{cooc}_{C}(h_i\rightarrow h_j)=\frac{|C_{h_i}\cap C_{h_j}|}{|C_{h_i}|}
$$
- Jaccard（対称）:
$$
J_C(h_i,h_j)=\frac{|C_{h_i}\cap C_{h_j}|}{|C_{h_i}\cup C_{h_j}|}
$$

head候補ベース共起の利点は、「同じ結論（同じ $(x,r_*,y)$）を複数ルールが支持している」状況を直接捉えられる点にある。これは (5) 支持構造（witness）の設計（複数根拠での支持を加点）と整合しやすい。

一方で、$C_h$ を得るには候補生成が必要であり、トリプル追加前のKG解析だけで完結する witnessベース共起より計算コストが高くなり得る。そのため初期arm生成では、次のいずれかの運用が現実的である。

- まず witnessベース共起で初期arm候補を絞り、絞り込んだルール群に対してのみ $C_h$ を生成して $\mathrm{cooc}_C$ を計算する
- もしくは、$r_*$ 周辺（例: $r_*$ を持つ head 集合）に対象を限定して $C_h$ を生成する

#### 2.2.2 実装での `cooc` の厳密定義（supported-target Jaccard）
本リポジトリで `initial_arms.json` に保存される `metadata.cooc` は、上の一般論のうち **「ターゲット集合 $\mathcal{T}_{\text{target}}$ に制限した $C_h$ のJaccard」**として実装されている。

まず、各ルール $h$ に対して「支持されるターゲットトリプル集合」
$$
\mathcal{S}(h)\;=\;\{\,t\in\mathcal{T}_{\text{target}}\mid \exists\,\theta:
  	heta\supseteq\theta_0(t,h)\;\wedge\;\mathcal{S}_{\text{cand}}\models B\,\theta\,\}
$$
を定義する。ここで

- $t=(x,r_*,y)$ は具体的なターゲット頭トリプル
- $\theta_0(t,h)$ は head パターンと $t$ の単一化（unification）で得られる初期代入
- $\mathcal{S}_{\text{cand}}\models B\,\theta$ は、候補集合上でbodyの全パターンが同時に充足されること（conjunctive query satisfiable）

実装上は [simple_active_refine/triples_editor.py](../../simple_active_refine/triples_editor.py) の
- `supports_head(t, h, TripleIndex(S_cand))`（存在判定）
- `count_witnesses_for_head(t, h, TripleIndex(S_cand), max_witness=...)`（witness数）
でこの条件を判定する。

次に、ルールペア $(h_i,h_j)$ の静的共起を
$$
\mathrm{cooc}(h_i,h_j)\;=\;\frac{|\mathcal{S}(h_i)\cap \mathcal{S}(h_j)|}{|\mathcal{S}(h_i)\cup \mathcal{S}(h_j)|}
$$
と定義する（Jaccard係数、対称）。この値が大きいほど「同じターゲットトリプルを説明できる（＝同一結論に対して冗長な支持を与え得る）」ことを意味する。

注意:
- `cooc=1.0` は「支持対象がほぼ同一」を意味し、**多様性の観点では冗長**になり得る。一方、(5) witness 構造（複数根拠での支持）と整合するため、初期シードとしては有効な場合がある。
- `cooc` は **初期arm候補の順位付け用の静的指標**であり、banditの報酬そのもの（受理数・witness合計・衝突ペナルティ等）ではない。

#### 2.2.3 初期arm生成アルゴリズム（実装準拠）
入力: body=2 ルール集合 $\mathcal{P}$、ターゲット集合 $\mathcal{T}_{\text{target}}$、候補集合 $\mathcal{S}_{\text{cand}}$、上位ペア数 $K$。

1. `TripleIndex` を構築し、$\mathcal{S}_{\text{cand}}$ に対するパターン照合を高速化する。
2. 各ルール $h\in\mathcal{P}$ について $\mathcal{S}(h)$ を計算する（各 $t\in\mathcal{T}_{\text{target}}$ で `supports_head` または `count_witnesses_for_head` を評価）。
3. すべてのペア $(h_i,h_j)$ について Jaccard により `cooc` を計算し、`cooc>0` のペアを候補として保持する。
4. `cooc` 降順でソートし、上位 $K$ 個を **pair-arm** として採用する（`metadata.kind="pair"`, `metadata.cooc=...`）。
5. すべてのルールを **singleton-arm** として採用する（`metadata.kind="singleton"`）。

#### 2.2.4 計算量・スケーリング（実装準拠）
記号:
- $M=|\mathcal{P}|$（ルール数）
- $T=|\mathcal{T}_{\text{target}}|$（ターゲット数）
- $N=|\mathcal{S}_{\text{cand}}|$（候補トリプル数）

1) インデックス構築
- `TripleIndex` は $O(N)$ で複数のハッシュ/辞書索引を構築する。

2) ルール支持判定（最重要）
- body=2 の conjunctive query に対する satisfiable 判定であり、最悪計算量は候補密度に依存する。
- 実装は `TripleIndex.match_pattern` を用いて、既に束縛された変数に基づく最も選択的な索引（(s,p)、(p,o) 等）を選ぶため、実運用では
  - 「第1パターンで得られる候補数」×「第2パターンの絞り込み」
  に近い。
- したがって総コストは概ね
$$
O\Big(\sum_{h\in\mathcal{P}}\sum_{t\in\mathcal{T}_{\text{target}}} \#\text{matches}(h,t)\Big)
$$
であり、$\#\text{matches}$ は `max_witness_per_head` により上限を課すことができる。

3) ペア `cooc` 計算
- 単純実装は $O(M^2)$ 個のペアに対して集合演算を行う。
- ただし各 $\mathcal{S}(h)$ は $T$ 個のターゲットトリプルの部分集合であり、集合の大きさが小さい場合（疎な支持）には実用的である。
- 現実的には $M$ を初期プールで絞る（上位数十〜数百）前提で運用する。

実務上の示唆:
- $T\le 10^4$ 程度であれば、ターゲット走査に基づく $\mathcal{S}(h)$ 構築は許容されやすい。
- $T$ が大きい場合、(i) ターゲットのサブサンプリング、(ii) head のみをキーにした近似、(iii) witnessベース共起で候補を絞ってからJaccard、などが必要になる。

### 2.3 反復精錬（k = 1..N）
反復 $k$ では、arm の選択→候補取得→評価/受理→履歴更新→KG更新を行う。

- **arm選択**: $\mathcal{A}$ から $k_{\text{sel}}$ 個の arm を選ぶ。方針は LLM / UCB / $\varepsilon$-greedy / Random を適用できるが、統計は「ルール」ではなく「arm」単位で保持する。

#### 2.3.1 LLM-policy（説明可能性 + 意味的整合性）
LLM-policy は、単に履歴の平均報酬を見るだけでなく、次の観測を入力として「次に選ぶarm」を判断する。

- **armの意味**: arm内ルールの body/head predicate と、その自然言語説明（`relation2text.txt` がある場合）
- **説明度（ターゲット整合）**: 直近の反復で「どれだけターゲットを説明したか」を示す diagnostics（例: targets coverage, witness統計）
- **証拠（evidence）**: 取得したevidenceトリプルのサンプル、および実際に追加されたevidenceトリプルのサンプル

これにより、定量的な「報酬」だけでなく、ターゲット関係の意味（例: nationalityなら地理/行政/居住を介する関係が有益になりやすい）に照らしてarmを選びやすくする。
- **候補取得**: 各 $a\in\mathcal{S}_k$ を適用して候補 $C_a$ を生成する。
  - 集合arm: 各ルールで生成した候補の和集合（重複は統合）
  - 順序付きarm: $h_{i_1}$ で得た **証拠（evidence/body）トリプル** を一時的なコンテキスト（例: in-memoryの候補集合や索引）へ追加し、その上で $h_{i_2}$ を適用…という逐次取得
    - 合意事項として、ターゲット関係 $r_*$（例: nationality）の **hypothesis トリプルは store-only** とし、反復中はKGへ確定追加しない（必要なら pending ストアに保持する）。
- **評価/受理（代理指標）**: 埋込スコアを用いず、次の指標で受理/棄却と報酬を定める。

#### (4) 機能的矛盾（衝突）
ターゲット関係 $r_*$ は 1対1 寄りとし、同一 head $x$ に対して複数の tail が立つことを衝突として扱う。

- 既存の $(x,r_*,y_{old})$ がある状態で $(x,r_*,y_{new})$ を追加する場合、
  - 原則として衝突（conflict）として扱い、強いペナルティ（報酬減点）または棄却を行う

ただし、ターゲット関係によっては例外（例: 複数国籍など）が存在し得る。そのため実装では、衝突判定を2段階に分ける。

1. **一次判定（ルールベース・高速）**:
   - 機能的制約に基づき機械的に衝突候補を列挙（例: 同一 $x$ に対して複数の $y$ が立つ）
   - 衝突候補を原則は棄却、または保留（pending）として隔離

2. **二次判定（LLM・任意）**:
   - 衝突候補について、既知のKG文脈（関連トリプル）や外部エビデンス（取得元の根拠）を入力として、
     「例外として両立し得るか」「どちらが妥当か」「追加を保留すべきか」をLLMに判定させる
   - 判定結果を受理/棄却に反映し、arm報酬にも反映する（例: 例外として受理した場合は減点を緩和）

#### (5) 支持構造（witness）
候補 $(x,r_*,y)$ に対して、arm 内の複数ルールがどれだけ厚く支持したかを報酬に用いる。

例（候補単位のスコア）:
$$
q(x,y)=\log(1+\sum_{h\in a} c_h(x,y))
$$

**relation priors によるKGE-friendly重み付け（推奨運用）**:

witness の寄与が「ハブ関係」や「KGEで扱いにくい関係」によって水増しされるのを抑制するため、
各述語 $p$ に prior $X_p\in[0,1]$ を与え、ルール $h$ の重みを

$$
W(h)=\prod_{p\in\mathrm{body\_predicates}(h)} X_p
$$

として、候補 witness を

$$
c'_h(x,y)=W(h)\,c_h(x,y)
$$

で置き換える（raw witness は観測値として保持し、proxy報酬には $c'_h$ を使う）。
詳細なI/Oと実装準拠の定義は [docs/rules/RULE-20260118-RELATION_PRIORS-001.md](RULE-20260118-RELATION_PRIORS-001.md) を参照。

arm の報酬は、受理候補に対する平均（または合計）から衝突ペナルティ等を差し引いて定義する。

例（arm報酬）:
$$
R(a)=\frac{1}{|C_a^{acc}|}\sum_{(x,y)\in C_a^{acc}} q(x,y)\; -\;\lambda\,\mathrm{conflicts}(C_a^{acc})\; -\;\mu\,\mathrm{hub\_bias}(C_a)
$$

- $\mathrm{hub\_bias}$ は人気ノード経由で witness が水増しされる現象を抑制するための正則化（例: 高次数ノードを含む witness を割引）。

備考:
- relation priors による重み付けは、$\mathrm{hub\_bias}$ の一部機能を「事前計算（オフライン）で吸収」する位置づけ。
- 反復内で KGE の再学習や再スコアリングを行わない、という設計制約と整合する。

- **履歴更新**: 選択した arm の報酬 $R(a)$ を履歴に記録し、次回選択の統計量（平均報酬、試行回数、直近平均など）を更新する。
- **KG更新**: 受理集合 $A_k=\bigcup_{a\in\mathcal{S}_k} C_a^{acc}$ を $T_{k-1}$ に追加して $G_k$ を得る。

- **arm拡張（任意）**: banditを回し始めた後、取得ログ（witness/衝突/候補集合の重なり など）を用いて「相性が良い」組み合わせを $\mathcal{A}$ に追加する。
  - 目的: 静的共起で取り切れない、ターゲット条件・探索過程・（順序付きarmの場合の）連鎖効果を取り込む。
  - 実装は任意（初期armのみでも動作する）。

### 2.4 再学習（バッチ）
反復ごとに再学習は行わず、以下のような条件でバッチ的に実施する。

- 追加トリプル数が閾値を超えた
- 衝突率が十分低い状態で一定回数反復した
- 固定スケジュール（例: K反復ごと）

最終的に $G_N$（または途中のチェックポイント）でKGEを再学習し、MRR/Hits@k を評価する。

補足（運用）:
- arm-run の反復自体は「埋込なし」で走らせ、後段でKGEの再学習（バッチ）を行う。
- 既に学習済モデルがある場合は、[retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py) を `--after_mode load` で実行し、before/after の評価のみ（再学習なし）を実施できる。
- 学習済 after が無い場合は、同スクリプトを `--after_mode retrain` で実行し、updated dataset で after を再学習してから評価できる。

## 3. 擬似コード（高レベル）
```python
# v3-combo-bandit: arm = combination of body=2 rules
current_kg = G0
rule_pool = extract_rules(current_kg)   # pool of body=2 rules
# pre-loop: singleton + co-occurring pairs (static, head-support based)
# inputs: target_triples (T_target), candidates (S_cand=train(+removed))
arms = build_initial_arms(rule_pool, target_triples, candidates, k_pairs=K)

for k in range(1, N+1):
    selected_arms = arm_selector.select(arms, k_sel, iteration=k)

    # acquire
    acquisition = acquire_with_arms(current_kg, selected_arms, iteration=k)
    # acquisition should include witness logs per rule/candidate

    # evaluate (proxy): witness + conflict (functional)
    evaluation = evaluator.evaluate(current_kg, acquisition, iteration=k)

    # history update (arm-level)
    arm_history.update(selected_arms, acquisition, evaluation)

    # expand arms using co-occurrence (optional)
    arms = maybe_expand_arms(arms, acquisition, evaluation, arm_history)  # optional

    # KG update
    current_kg.add(evaluation.accepted_triples)

# periodic or final re-train
final_result = kge_trainer.train_and_evaluate(current_kg)
```

## 4. 実装メモ（v3資産の活用）
- v3のパイプライン構造（selector/acquirer/evaluator/history）を温存し、
  - selector/history の単位を rule から arm に拡張
  - evaluator を「件数報酬」から「witness+衝突」へ差し替え
  - acquirer に witness ログ（置換数）を返す機構を追加
 という差分実装で導入できる。

- 参考実装箇所（v3）:
  - 反復ループ: [simple_active_refine/pipeline.py](../../simple_active_refine/pipeline.py)
  - 候補取得: [simple_active_refine/triple_acquirer_impl.py](../../simple_active_refine/triple_acquirer_impl.py)
  - bodyマッチング: [simple_active_refine/triples_editor.py](../../simple_active_refine/triples_editor.py)
  - 評価器差し替え: [simple_active_refine/triple_evaluator_impl.py](../../simple_active_refine/triple_evaluator_impl.py)
  - 履歴: [simple_active_refine/rule_history.py](../../simple_active_refine/rule_history.py)

### 4.1 初期arm生成の実装（本派生で追加）
- Armデータ構造: [simple_active_refine/arm.py](../../simple_active_refine/arm.py)
- 初期armビルダ（supported-target Jaccard）: [simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py)
  - `ArmBuilderConfig.k_pairs`: 上位ペアarm数
  - `ArmBuilderConfig.max_witness_per_head`: `count_witnesses_for_head` の打ち切り（計算量制御）
- CLI: [build_initial_arms.py](../../build_initial_arms.py)
  - `--rule-pool`: `initial_rule_pool.pkl`
  - `--target-triples`: `target_triples.txt`
  - `--dir-triples` + `--include-train-removed`: `train.txt`（+任意で`train_removed.txt`）を候補集合としてロード
  - 出力: `initial_arms.json` / `initial_arms.pkl` / `initial_arms.txt`

### 4.2 `initial_arms.json` の読み方（再現性のための仕様）
- 各要素は arm であり、以下のフィールドを持つ。
  - `arm_type`: 現状は `"set"` のみ（順序付きarmは将来拡張）
  - `rule_keys`: 文字列化されたルール（現状は `str(AmieRule)` をキーとして使用）
  - `metadata.kind`: `"singleton"` または `"pair"`
  - `metadata.cooc`: `kind=="pair"` の場合のみ。2.2.2 の Jaccard 定義に一致。

注意（実務上の落とし穴）:
- `rule_keys` を `str(rule)` にしているため、将来的に `__str__` 表現が変わるとキーが不安定になり得る。履歴の永続化・比較実験の再現性の観点では、ルールID（ハッシュ）を別途導入するのが望ましい。
