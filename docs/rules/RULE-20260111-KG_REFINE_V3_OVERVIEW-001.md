# RULE-20260111-KG_REFINE_V3_OVERVIEW-001: ルール駆動KGリファイン (v3) 概要

作成日: 2026-01-11
移管元: [docs/algorithm_overview_v3.md](../algorithm_overview_v3.md)

---

## 1. アルゴリズムの概要
- 目的: 既存KGに外部情報を統合しつつ、Hornルールに基づく追加トリプルを段階的に採用し、再学習を繰り返すことでリンク予測性能とカバレッジを向上させる。
- 着想: (a) 埋込モデルで高信頼トリプル近傍を抽出しAMIE+でHornルールを得る、(b) ルールに基づき候補トリプルを生成・受理しKGを拡張、(c) 拡張後KGで再学習してスコア分布を更新する、という反復最適化。
- 特徴: ルールプールを初回に構築し、以降は固定プールを効率再利用。ルール選択は履歴駆動（LLM/UCB/ε-greedy/Random）で行い、候補取得は「取得＝受理」がデフォルト。評価器は任意で挿入可能（フィルタ/スコア付け）。

## 2. アルゴリズムの詳細
記法: KGを $G_k=(E,R,T_k)$、ターゲット関係を $r_*$、Hornルールを $h: B(x,y,\mathbf{z}) \Rightarrow (x,r_*,y)$ とする。反復回数を $N$ とし、$k$-回目のKGを $G_k$ と表記する。ルールプールを $\mathcal{P}$、選択集合を $\mathcal{S}_k$、候補を $C_h$、受理集合を $A_k$ とする。

最適化目標（概念図）: 追加トリプル集合 $A_k$ を通じて最終KGEの指標 $\mathrm{MRR}(G_N)$ を高めつつ、冗長・誤りトリプルを抑制すること。単純化した目的関数は、$\max \; \mathbb{E}[\mathrm{MRR}(G_N)] - \lambda \sum_k |A_k^{\text{err}}|$ とみなせる（$A_k^{\text{err}}$ は誤りトリプル集合、$\lambda$ はペナルティ係数）。実装ではこの目的を近似するために、ルール報酬と履歴に基づくバンディット的探索を行う。

### 2.1 初期ルールプール構築 (イテレーション0 相当)
- 埋込学習: $T_0$ でKGEを学習し、トリプルスコア $s(t)\in[0,1]$ を得る。
- 高スコア抽出: 上位パーセンタイル $p$ の集合 $T^{\text{hi}}=\{t\mid s(t)\ge q_p(s)\}$ を抽出し、$k$-hop 近傍サブグラフ $G^{\text{sub}}$ を生成。
- ルール抽出: AMIE+を $G^{\text{sub}}$ に適用し Horn ルール集合 $\mathcal{H}$ を得る。品質指標 $(c_{\text{PCA}}, c_{\text{head}}, |B|)$ を取得。
- ルールスコア: $\phi(h)=0.4\,\hat c_{\text{PCA}}+0.3\,\hat c_{\text{head}}+0.3\,\hat d$ を計算（$\hat d=1/(|B|+1)$）。トップ $m$ をプール $\mathcal{P}$ に格納。LLMフィルタを使う場合は $\mathcal{H}$ を LLM でスコアし上位 $m$ を採用。

### 2.2 反復精錬 (k = 1..N)
- **ルール選択**: $\mathcal{P}$ から $k_{\text{sel}}$ 本を選ぶ。初回（iteration=1）はプール順を優先し、以降は履歴統計に基づき LLM / UCB / $\varepsilon$-greedy / Random で選択。履歴統計は `RuleHistory` の平均報酬、試行回数、直近平均などを用いる。
- **候補取得**: 各 $h\in\mathcal{S}_k$ について、ターゲットトリプルをサンプルしbodyを充足する置換を探索、候補 $C_h$ を生成。ベースラインとしてルール無視のランダム取得、拡張としてWeb/LLM検索をサポート。
- **履歴更新・受理**: デフォルトでは $C_h$ をそのまま受理（暗黙受理）。評価器を指定すれば受理/棄却やスコア付けを差し替え可能。受理集合 $A_k=\bigcup_h C_h$（フィルタありなら $C_h^{\text{accepted}}$）を KG に追加し $G_k = G_{k-1} \cup A_k$。報酬は件数等で算出し RuleHistory に記録。

### 2.3 最終埋込学習
- $G_N$ でKGEを再学習し、Hits@1/3/10, MRR を計測。モデルと指標を保存。

### 2.4 擬似コード
```python
# 高レベル擬似コード（pipeline.run に整合）
current_kg = G0
rule_pool = extract_rules(current_kg)  # iteration=0 相当で構築
for k in range(1, N+1):
   # ルール選択
   selected = rule_selector.select(rule_pool, k_sel, iteration=k) if rule_selector else rule_pool

   # 候補取得
   acquisition = triple_acquirer.acquire(current_kg, selected, iteration=k)

   # 評価/受理（デフォルトは暗黙受理）
   if evaluator:
      evaluation = evaluator.evaluate(current_kg, acquisition, iteration=k)
   else:
      evaluation = implicit_accept(acquisition)  # 件数を報酬に

   # 履歴更新
   rule_history.update(selected, acquisition, evaluation)

   # KG更新
   current_kg.add(evaluation.accepted_triples)

final_result = kge_trainer.train_and_evaluate(current_kg)
```

## 2. アルゴリズムの詳細
記法: 初期KGを $G_0=(E,R,T_0)$、ターゲット関係を $r_*$、Hornルールを $h: B \Rightarrow (x,r_*,y)$ とする。反復回数を $N$ とし、$k$-回目のKGを $G_k$ とする。

1. **初期ルールプール構築 (イテレーション0)**
   - 埋込モデルで $T_0$ を学習し、各トリプルのスコア $s(t)\in[0,1]$ を得る。
   - 上位パーセンタイル $p$ のトリプル集合 $T^{\text{hi}}=\{t\mid s(t)\ge q_p(s)\}$ を取り、$k$-hop 近傍サブグラフ $G^{\text{sub}}$ を構築。
   - AMIE+で $G^{\text{sub}}$ から Horn ルール集合 $\mathcal{H}$ を抽出し、品質指標 $(c_{\text{PCA}}, c_{\text{head}}, |B|)$ を得る。
   - ルール選択スコア $\phi(h)=0.4\,\hat c_{\text{PCA}}+0.3\,\hat c_{\text{head}}+0.3\,\hat d$ を計算（$\hat\cdot$ は0-1正規化、$\hat d=1/(|B|+1)$）。トップ $m$ 件をプール $\mathcal{P}$ とする。LLMスコアリングを用いる場合は $\mathcal{H}$ を LLM でランク付けし、上位 $m$ を $\mathcal{P}$ とする。

2. **反復精錬 (k=1..N)**
   - **ルール選択**: プール $\mathcal{P}$ から $k_{\text{sel}}$ 本を選ぶ。方針は複数あり、履歴をLLMに渡して構造化選択するもの、UCBで未試行にボーナスを与えるもの、$\varepsilon$-greedyで探索と活用を混在させるもの、完全ランダムに選ぶものがある。選択結果と得られた報酬は履歴に蓄積し、次ラウンドの選択に利用する。
   - **候補取得**: 選ばれた各ルールについて、ターゲット三つ組から少数サンプルしてルールのbodyを満たす置換を探し、候補トリプル集合 $C_h$ を作る。ベースラインとして、ルールを無視して候補プールからランダムに抽出する手法も併用可能。
   - **履歴更新（報酬計算）**: デフォルトでは「候補取得で得られた $C_h$ をそのまま受理」とし（= 候補取得が受理判定を兼ねる）、ルール報酬を件数ベースで付与して履歴を更新する。必要に応じて、外部エビデンスやスコア変化を用いたフィルタ（受理/棄却）を“任意で”挿入できる。受理集合 $A_k=\bigcup_h C_h$（フィルタありの場合は $A_k=\bigcup_h C_h^{\text{accepted}}$）を KG に追加し $G_k = G_{k-1} \cup A_k$ とする。

3. **最終埋込学習**
   - $G_N$ で埋込モデルを再学習し、指標 (Hits@1/3/10, MRR) を算出。モデルパラメータ $\theta^{\text{final}}$ を保存。

## 3. 各ステップの詳細

### 3.1 ルール選択

#### 概要
固定プール $\mathcal{P}$ から、各反復で使用するルール集合 $\mathcal{S}_k$（サイズ $k_{\text{sel}}$）を選ぶ。目的は、限られた反復回数の中で「有望なルールの活用」と「未知ルールの探索」をバランスさせ、改善に寄与するルールへ試行回数を配分すること。

#### アルゴリズム
選択器は、(i) ルール品質（初期指標など）、(ii) 過去の試行結果（報酬）のいずれか／両方を用いて $\mathcal{S}_k \subseteq \mathcal{P}$ を決める。

- **LLMポリシー型**: 履歴統計（試行回数、平均報酬、直近傾向など）を入力として、LLMが選択方針と $k_{\text{sel}}$ 個のルールIDを構造化出力する。
- **UCB型**: 各ルール $h$ に対し $\text{UCB}(h)=\mu(h)+c\sqrt{\ln n / n(h)}$ を計算し上位を選ぶ（$\mu$ は平均報酬、$n$ は全試行、$n(h)$ は当該ルールの試行数）。未試行は探索優先として扱う。
- **$\varepsilon$-greedy型**: 確率 $\varepsilon$ で探索（ランダム選択）、$1-\varepsilon$ で活用（推定報酬が最大のものから選択）。
- **ランダム型**: 一様ランダムに $k_{\text{sel}}$ 個を選ぶ（比較用ベースライン）。

報酬の既定形: 受理トリプル数（または受理率）を実数に変換して付与。必要に応じてスコア変化や外部評価を組み込める。履歴統計には平均報酬、標準偏差、直近平均、成功率（正のスコア変化割合）を保持する。

#### 実装
- セレクタ本体: [simple_active_refine/rule_selector.py](simple_active_refine/rule_selector.py)（`LLMPolicyRuleSelector` / `UCBRuleSelector` / `EpsilonGreedyRuleSelector` / `RandomRuleSelector`）。初回はプール順を尊重し、以降は履歴ベースで選択。
- 履歴（報酬・統計）: [simple_active_refine/rule_history.py](simple_active_refine/rule_history.py) が `RuleHistory`/`RuleEvaluationRecord` を管理し、セレクタは `history.get_all_rule_statistics()` を参照。
- 反復ループ: [simple_active_refine/pipeline.py](simple_active_refine/pipeline.py#L110-L199) で `rule_selector.select_rules()` を呼び、選ばれたルールのみを候補取得に渡す。

```python
# pipeline.run 抜粋
if self.rule_selector:
   k = self.n_select_rules or len(rule_pool_with_id)
   selected_with_id, _ = self.rule_selector.select_rules(rule_pool_with_id, k=k, iteration=iteration)
   selected_rules = [rwi.rule for rwi in selected_with_id]
else:
   selected_with_id = rule_pool_with_id
   selected_rules = extracted.rules
```

### 3.2 候補取得

#### 概要
選択されたルール $h\in\mathcal{S}_k$ を用いて、追加候補トリプル集合 $C_h$ を生成する。候補取得は「ルールbodyを満たす観測（内部候補プールや外部エビデンス）」を探索し、head（ターゲット関係）の追加候補へ変換するプロセス。デフォルト実装は内部プールを用いたルール駆動取得で、ランダム基準やWeb/LLM検索への切り替えも可能。

#### アルゴリズム
基本形は、(1) ターゲットトリプル（検証対象）のサンプル、(2) bodyの充足（マッチング／探索）、(3) head候補生成、の3段。

- **ルール駆動（内部プール）**: `RuleBasedTripleAcquirer` がターゲットトリプルをサンプルし、`add_triples_for_single_rule` で body 一致を確認して head 候補 $C_h$ を生成。`reuse_targets` でターゲットの再利用を制御。
- **ランダム（ベースライン）**: `RandomTripleAcquirer` がルールを無視して候補プールから一様サンプリングし、取得フェーズの寄与を切り分ける。
- **外部検索（拡張）**: `WebSearchTripleAcquirer` がターゲットトリプルを種に LLM+検索でエビデンスを取得し、body 整合するものを候補化する。

#### 実装
- 候補取得実装: [simple_active_refine/triple_acquirer_impl.py](simple_active_refine/triple_acquirer_impl.py) に `RuleBasedTripleAcquirer`（train_removed をbodyマッチング）/`RandomTripleAcquirer`（ルール無視）/`WebSearchTripleAcquirer`（LLM+検索）。
- bodyマッチング: [simple_active_refine/triples_editor.py](simple_active_refine/triples_editor.py) の `add_triples_for_single_rule()` が Horn body を充足する置換を求め、head候補を返す。
- パイプライン連携: [simple_active_refine/pipeline.py](simple_active_refine/pipeline.py#L191-L207) で `triple_acquirer.acquire()` を呼び、ルール別 `candidates_by_rule` を得る。

```python
# RuleBasedTripleAcquirer.acquire 抜粋
for rule in context.rules:
   available = target_triples if reuse_targets else [t for t in target_triples if t not in used_targets]
   sample_size = min(n_targets_per_rule, len(available))
   sampled = random.sample(available, sample_size)
   added_triples, _ = add_triples_for_single_rule(...)
   if added_triples:
      candidates_by_rule[str(rule)] = [tuple(t) for t in added_triples]
```

### 3.3 ルール評価

#### 概要
候補集合 $C_h$ に基づいて、ルール $h$ の報酬（次回以降の選択の根拠）を更新する。v3ではデフォルトで「候補取得が受理判定を兼ねる」ため、評価は主に報酬付与と履歴更新に焦点を当てる。必要に応じて候補のフィルタリング（受理/棄却）を“任意で”追加できる。

#### アルゴリズム
- **受理/棄却（任意）**: デフォルトは $C_h$ を全受理（暗黙受理）。必要なら、外部エビデンス整合・スコア変化・閾値判定等によるフィルタで $C_h^{\text{accepted}}$ を得る。
- **報酬設計**: 追加数（$|C_h|$ または $|C_h^{\text{accepted}}|$）、受理率、推定スコア改善、対象トリプルの改善量などを報酬として定義できる。

#### 実装
- デフォルトは evaluator なし: [simple_active_refine/pipeline.py](simple_active_refine/pipeline.py#L199-L234) で acquisition をそのまま受理し、件数を報酬として `TripleEvaluationResult` を自動生成。
- フィルタを入れたい場合: [simple_active_refine/triple_evaluator_impl.py](simple_active_refine/triple_evaluator_impl.py) の `AcceptAllTripleEvaluator` など任意の evaluator を指定可能（受理/棄却やスコア付与を差し替え）。
- 履歴更新: [simple_active_refine/pipeline.py](simple_active_refine/pipeline.py#L236-L275) が evaluator の出力を基に `RuleHistory.add_record()` を呼び、報酬・件数・スコア統計を記録。

評価器を入れた場合の挙動: evaluator が返す `accepted_triples` / `rejected_triples` / `rule_rewards` / `triple_scores` がそのまま履歴更新・KG更新に使用される。外部エビデンスを用いるフィルタやスコア変化ベースの報酬へ差し替え可能。

```python
# evaluator 省略時の暗黙受理 (pipeline.run)
accepted, rule_rewards = [], {}
for rule_key, triples in acquisition.candidates_by_rule.items():
   accepted.extend(triples)
   rule_rewards[rule_key] = float(len(triples))
unique = list(dict.fromkeys([tuple(t) for t in accepted]))
evaluation = TripleEvaluationResult(
   accepted_triples=unique,
   rejected_triples=[],
   rule_rewards=rule_rewards,
   triple_scores={t: 1.0 for t in unique},
   diagnostics={"n_accepted": len(unique), "implicit_acceptance": 1.0},
)
```

## 4. 実装

### 4.x 補足（派生: armベース運用の前処理）

本ドキュメントは v3 の「ruleベース反復精錬」を中心に記述しているが、派生として arm（ルール組）を単位に選択する運用（combo-bandit）がある。

arm生成は「singleton arm = ルール数」となるため、プールが大きいと探索が分散し、短い反復（例: 20iter）では同一armの再選択が起きにくい。
この問題に対しては、arm生成前に rule_pool を品質指標（pca_conf/head_coverage等）またはLLMで事前フィルタし、arm数を抑えるのが有効。

参照:
- [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- [docs/records/REC-20260111-ARM_REFINE-005.md](../records/REC-20260111-ARM_REFINE-005.md)

- オーケストレーション: [simple_active_refine/pipeline.py](simple_active_refine/pipeline.py) の `RuleDrivenKGRefinementPipeline` が抽象インタフェースを定義し、反復処理と最終KGE学習を管理。
- メインスクリプト: [main_v3.py](main_v3.py) が設定読み込み、初期KG・ターゲット読み込み、初回ルールプール構築後にパイプラインを実行。
- ルール抽出: [simple_active_refine/rule_extractor_impl.py](simple_active_refine/rule_extractor_impl.py) の `HighScoreRuleExtractor` が KGE 学習→高スコアサブグラフ→AMIE+→多様性スコア/LLMフィルタ→プール化を実施。以降は `PrecomputedRuleExtractor` がプールを再利用。
- 候補生成: [simple_active_refine/triple_acquirer_impl.py](simple_active_refine/triple_acquirer_impl.py) の `RuleBasedTripleAcquirer` が各ルールにつきターゲットをサンプルし、body マッチングで候補を生成。ベースラインとして `RandomTripleAcquirer` も用意。
- 評価（任意）: デフォルトは「候補取得 = 暗黙受理」で、評価器は指定しない。フィルタや追加のスコア付けを行いたい場合は [simple_active_refine/triple_evaluator_impl.py](simple_active_refine/triple_evaluator_impl.py) の評価器（例: `AcceptAllTripleEvaluator` など）を指定して差し替える。
- 埋込再学習: [simple_active_refine/kge_trainer_impl.py](simple_active_refine/kge_trainer_impl.py) の `FinalKGETrainer` が最終KGで KGE を学習し、指標とモデルを出力。
- データ管理: [simple_active_refine/data_manager.py](simple_active_refine/data_manager.py) が各イテレーションのトリプル書き出しとカスタムデータセット生成を担当。
