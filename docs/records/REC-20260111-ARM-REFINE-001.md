# REC-20260111-ARM-REFINE-001: 反復精錬（armベース）実装計画

作成日: 2026-01-11

## 0. 目的と前提

### 目的（この計画で実装する範囲）
本プロジェクトの v3-combo-bandit 方針に基づき、**ルールではなくarm（ルール組み合わせ）を選択単位**として反復精錬を実装する。具体的には、[docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md) の **2.3 反復精錬**に相当する「選択→取得→評価→履歴更新→KG更新」の反復ループを、既存の v3 パイプライン資産を最大限再利用しつつ実装する。

### 前提（すでに完了しているもの）
- 初期ルールプール生成: [build_initial_rule_pool.py](../../build_initial_rule_pool.py)
- 初期arm候補生成（singleton + pair, cooc=Jaccard）: [build_initial_arms.py](../../build_initial_arms.py), [simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py)
- witness/support 判定ユーティリティ: [simple_active_refine/triples_editor.py](../../simple_active_refine/triples_editor.py)

### 重要な設計制約
- **反復中はKGEの再学習やスコア計算を必須にしない**（代理指標中心）。
- 既存の抽象化（pipeline/selector/history）を可能な限り再利用する。
- ただし、`TriplesFactory.from_path` をルールごとに叩くような明確な性能劣化がある場合は、**新規実装（高速化）を優先**する。

### 取得ソースに関する追加要望（2026-01-11）
- 検証段階のトリプル取得は `train_removed.txt`（ローカル候補集合）を主とする。
- ただし将来的に [simple_active_refine/triple_acquirer_impl.py](../../simple_active_refine/triple_acquirer_impl.py) の `WebSearchTripleAcquirer` 相当の **Web検索/LLM取得**も差し替え可能にしたい。
- そのため本計画では、取得（Acquirer）を **「バックエンド差し替え可能」**な設計に寄せ、同一の arm パイプラインで `local|web` を切り替えられるようにする。

### 方針決定（2026-01-11 合意事項）
- 反復でKGに追加する主対象は **nationality（$r_*$）ではなく、ルールbody側の「証拠（evidence）トリプル」**とする。
  - 例: place_of_birth / places_lived / film.country / location.contains 等。
  - 狙い: KGの周辺構造を厚くし、後段（再学習KGEや最終LLM判断）で nationality の整合性・推定精度を上げる。
- nationality（$r_*$）トリプルの追加は「主」にはしない。
  - 必要であれば **候補（hypothesis）** として生成・保持は許可するが、KGへの確定追加は評価器/LLM判定のフェーズに委ねる。
  - **当面は store-only**（hypothesis は「保留ストア」に保持し、KGには追加しない）。
- conflict（同一headに複数tail）の扱い:
  - **複数を許可**し、一次判定では「矛盾候補（pending）」として保持する。
  - **最終的にはLLMで判断**（受理/棄却/保留）する。

## 1. 既存コードの再利用方針

### 1.1 再利用できる資産（そのまま使う）
- 反復ループの抽象化とコンテキスト/結果型: [simple_active_refine/pipeline.py](../../simple_active_refine/pipeline.py)
- ルール選択戦略（UCB/ε-greedy/LLM-policy など）の設計パターン: [simple_active_refine/rule_selector.py](../../simple_active_refine/rule_selector.py)
- 履歴の設計パターン: [simple_active_refine/rule_history.py](../../simple_active_refine/rule_history.py)
- パターン照合（unification + backtracking）と索引: [simple_active_refine/triples_editor.py](../../simple_active_refine/triples_editor.py)
  - `TripleIndex`, `supports_head`, `count_witnesses_for_head`, `find_body_triples_for_head`

### 1.2 置き換え/追加が必要な箇所
- 選択単位が rule → arm に変わるため、以下を arm 用に新設する。
  - ArmSelector（RuleSelector相当）
  - ArmHistory（RuleHistory相当）
  - ArmDrivenPipeline（RuleDrivenKGRefinementPipeline相当）
- 取得・評価も arm 単位に拡張する。
  - 取得: arm（=複数ルール）をまとめて処理し、候補トリプル・witnessログを生成
  - 評価: witness + conflict（機能的制約）など、代理指標に基づき受理/棄却と報酬を計算

## 2. 実装アーキテクチャ（提案）

### 2.1 新規/拡張する主要コンポーネント

#### (A) armの識別子と永続性
- 現状の arm 表現: [simple_active_refine/arm.py](../../simple_active_refine/arm.py)
  - `Arm.key()` が deterministic key を提供。
- 反復精錬では **arm_id を安定化**したい。
  - 方針: `arm_id = "arm_" + hash(Arm.key())` のように、arm内容から決定するIDを導入。
  - 目的: 実験の再現性、履歴の保存・比較、arm集合の拡張時の重複排除。

#### (B) ArmHistory（統計・レポート）
- 既存の [simple_active_refine/rule_history.py](../../simple_active_refine/rule_history.py) を踏襲し、arm版を新設。
- 記録単位（案）:
  - `ArmEvaluationRecord(iteration, arm_id, arm, selected_targets, acquired_candidates, accepted_triples, rejected_triples, reward, diagnostics, policy_text(optional))`
- 統計（案）:
  - trials, mean_reward, std_reward, acceptance_rate, conflict_rate, recent_performance
- 保存形式:
  - pickle + JSON（既存と同様）

#### (C) ArmSelector（選択戦略）
- 既存の [simple_active_refine/rule_selector.py](../../simple_active_refine/rule_selector.py) を可能な限り再利用する。
- 方針:
  - `create_arm_selector(strategy=...)` を追加し、`ucb|epsilon_greedy|llm_policy|random` を提供。
  - LLM-policy の場合は、入力を rule統計 → arm統計に置き換えるだけでプロンプト雛形を再利用。

#### (D) ArmTripleAcquirer（arm単位の候補取得）
armは複数ルールを含むため、取得は「ルールごと」ではなく「armごと」の結果が必要。

- 既存の候補取得の入口: [simple_active_refine/triple_acquirer_impl.py](../../simple_active_refine/triple_acquirer_impl.py)
  - `RuleBasedTripleAcquirer` は `add_triples_for_single_rule` を呼ぶが、内部で `TriplesFactory.from_path` を毎回実行し得るため重い。
  - `WebSearchTripleAcquirer` は「target_triples（seed）→ web/LLM → triples」を返す設計で、**将来の取得バックエンド**として活用したい。

- 提案: arm版では「取得バックエンド（local/web）」を明確に分離する
  - `ArmTripleAcquirer` は arm を入力に **候補トリプル集合（および証拠ログ）**を返す“統一I/F”にする。
  - 実体は `LocalArmTripleAcquirer`（train/train_removed）と `WebArmTripleAcquirer`（LLM/web）に分け、設定で切り替える。
  - さらに `CompositeArmTripleAcquirer` を用意し、`local` を主、`web` をフォールバック/追加入手として合成できるようにする（将来拡張、初期は単独でもOK）。

- 提案する高速実装（新設）: LocalArmTripleAcquirer
  - 反復ごとに一度だけ `TripleIndex(S_cand)` を構築し、ルールbody照合を高速化。
  - `target_triples` は反復内でサンプリング（n_targets_per_arm）可能。
  - ルール適用は `find_body_triples_for_head` を用いて「必要な証拠（body triples）」を抽出。
  - 併せて witness ログ（`count_witnesses_for_head`）を計測し、評価器に渡す。

- Web検索バックエンド（将来有効化）: WebArmTripleAcquirer
  - 目的: ローカル候補集合に無い証拠（evidence）を外部から補う。
  - seed 設計（nationalityを主にしない方針に整合）:
    - seed entity: target triple の head（例: person）
    - seed relations: arm内ルールの body に現れる述語（例: film.country, location.contains など）
    - これにより `r_*`（nationality）を直接問い合わせるより、証拠側の構造を取りにいける。
  - 取得結果は provenance（URL/スニペット/検索クエリ等）を保持し、KGへ加える “evidence triples” に紐付けてダンプする。
  - Web取得した triples を（必要なら）一時的な `TripleIndex` に入れて同様の body 照合を行い、armが要求する形の evidence を抽出する。

- 取得物の区別（重要）:
  - **evidence triples**: bodyを満たすために見つかった（または必要となる）トリプル。反復でKGへ追加する主対象。
  - **hypothesis triples（任意）**: $(x,r_*,y)$ 形式の候補。必要なら生成しても良いが、確定追加はLLM/評価により制御する。

- 取得結果に provenance を載せる（web対応のため）
  - 最小設計: `Triple` 自体は `(h,r,t)` のまま維持し、別途 `provenance_by_triple: Dict[Triple, Dict]` を返す。
  - `local` の provenance は `{source: "train_removed"|"train" , file: ..., iteration: ...}` 程度。
  - `web` の provenance は `{source: "web", query: ..., urls: [...], excerpt: ..., retrieved_at: ...}` 等（詳細は後で詰める）。

- 取得結果の構造（案）:
  - `candidates_by_arm: Dict[arm_id, List[Triple]]`
  - `witness_by_arm_and_target: Dict[arm_id, Dict[target_triple, int]]` など
  - 既存の `TripleAcquisitionResult` を拡張するか、arm専用 Result 型を新設する（後述）。

#### (E) Witness+Conflict Evaluator（代理評価器）
- 現状: [simple_active_refine/triple_evaluator_impl.py](../../simple_active_refine/triple_evaluator_impl.py) は AcceptAll のみ。
- 新設する評価器（案）:
  - 受理判定:
    - 既存KGにすでにあるトリプルは除外
    - 機能的制約（同一headに複数tailが立つ等）による衝突は **棄却せず**、pendingとして保持（複数許可）
    - pending の最終判定は LLM に委譲（受理/棄却/保留）
  - 報酬:
    - witness（支持の厚み）と accepted数を加点
    - conflict数を減点

- 実装方針:
  - conflictはターゲット関係 r* に対してのみ厳密に適用（例: nationality）
  - 「例外（複数国籍など）」を扱う LLM 判定はオプション（後回しでOK）

### 2.2 パイプライン統合案（最小改変）

#### 案1: arm専用パイプラインを新設（推奨）
- `RuleDrivenKGRefinementPipeline` をコピー・簡素化して `ArmDrivenKGRefinementPipeline` を新設。
- 利点:
  - 既存ruleパイプラインと干渉しない
  - arm固有の I/O（witness, conflict, candidates_by_arm）を自然に扱える
- 追加ファイル例:
  - `simple_active_refine/arm_pipeline.py`
  - `simple_active_refine/arm_selector.py`
  - `simple_active_refine/arm_history.py`
  - `simple_active_refine/arm_triple_acquirer_impl.py`
  - `simple_active_refine/arm_triple_evaluator_impl.py`

#### 案2: 既存pipelineを汎用化（非推奨・作業量大）
- `RuleDrivenKGRefinementPipeline` を「選択単位」を抽象化して共通化する。
- 利点: 重複コード削減
- 欠点: 既存の `RuleSelector/RuleHistory/TripleAcquisitionResult` との整合が難しく、リスクが高い。

結論: まず案1で動くものを作り、必要が出たら共通化を検討。

## 3. 実装ステップ（マイルストーン）

### M1: armの永続ID化 + ArmHistory
- 追加/変更
  - `ArmWithId.create(...)` に deterministic な arm_id を付与できるようにする（または新メソッド追加）
  - `ArmHistory` を新設（保存・統計・Markdownレポート）
- 検証
  - 同じarm入力から同じarm_idが生成される
  - 統計計算（mean/std/recent）が期待通り

### M2: ArmSelector
- 追加
  - `ArmSelector` 基底 + `UCBArmSelector` / `EpsilonGreedyArmSelector` / `LLMPolicyArmSelector` / `RandomArmSelector`
  - `create_arm_selector` ファクトリ
- 再利用
  - [simple_active_refine/rule_selector.py](../../simple_active_refine/rule_selector.py) の実装パターンを踏襲
- 検証
  - 乱数seed固定で選択が再現可能
  - UCBが未試行armを優先する

### M3: 高速 ArmTripleAcquirer（train/train_removedベース）
- 追加
  - `ArmBasedTripleAcquirer` を「I/F（arm→候補 + witness + provenance）」として新設
  - 実装1（検証段階）: `LocalArmTripleAcquirer`（train/train_removed）
  - 実装2（将来）: `WebArmTripleAcquirer`（LLM/web検索）
  - 実装3（任意）: `CompositeArmTripleAcquirer`（local主 + web追加入手の合成）
- 重要な性能設計
  - 反復ごとに `TripleIndex` を1回だけ構築
  - `max_witness_per_head` と `n_targets_per_arm` による上限
  - I/Oダンプ（必要に応じて）
- 検証
  - 小規模データで既存 `RuleBasedTripleAcquirer` と同等の候補が得られる（サンプリング条件が同じ場合）
  - （Web実装を有効化した場合）provenance がダンプされ、同一seedでキャッシュが効く

### M4: Proxy Evaluator（witness + conflict）
- 追加
  - `WitnessConflictTripleEvaluator`（arm単位の報酬・受理/棄却・診断）
- 方針
  - conflictはターゲット関係のみに適用（設定で切替）
  - conflictは一次で棄却せず pending として保持
  - LLMによる最終判定をM6（後続）で実装可能な形にインターフェースを設計
- 検証
  - 人工ケースで衝突が検出できる
  - witnessが大きいほど報酬が上がる

### M5: ArmDrivenKGRefinementPipeline + 実行スクリプト
- 追加
  - `ArmDrivenKGRefinementPipeline`（選択→取得→評価→履歴更新→KG更新）
  - CLI例: `run_arm_refinement.py`（実験ディレクトリを入力として iter_1.. を生成）
- 再利用
  - [simple_active_refine/pipeline.py](../../simple_active_refine/pipeline.py) の `RefinedKG` と run構造
  - [simple_active_refine/data_manager.py](../../simple_active_refine/data_manager.py) の出力管理（可能なら）
- 成果物
  - `experiments/<date>/.../iter_k/` に
    - `accepted_triples.json`
    - `arm_history.pkl/json`
    - `report.md`（簡易でOK）

### M6: 追加最適化（必要に応じて）
- LLMによる conflict例外判定（オプション）
- arm拡張（ログから新ペア生成）
- ルールIDの安定化（`str(rule)` 依存の排除; ハッシュID導入）

## 4. 性能・スケーリング設計（必須観点）

### 4.1 ボトルネック仮説
- `add_triples_for_single_rule` の内部で毎回 `TriplesFactory.from_path` を実行するパスは、反復・arm化で顕著に遅くなる可能性が高い。
- arm化により「1反復あたりに評価するルール数」が増えるため、body照合は `TripleIndex` ベースに寄せる。

### 4.2 採用する最適化
- 候補集合の索引（`TripleIndex`）を反復内で共有
- witness計数の打ち切り（`max_witness_per_head`）
- ターゲットのサンプリング（`n_targets_per_arm`）
- 可能なら、ルールごとのパターン順序を「選択性の高い順」に並べ替える（将来拡張）

### 4.3 Web検索を見越した運用設計（有効化時）
- キャッシュ:
  - key例: `(entity_id, relation_id, iteration, model_version)` を基本にメモ化し、同一seedの重複呼び出しを抑止。
  - 生の取得結果（レスポンスJSON）も実験ディレクトリに保存し、再実行時に再利用可能にする。
- レート制御と失敗耐性:
  - 並列数・リトライ回数・指数バックオフを設定化。
  - 失敗しても反復全体を落とさず、localのみで続行できる。
- provenance の保持:
  - 受理された evidence triple は provenance とセットで `accepted_triples.json` に記録（後段のLLM判断・監査のため）。

## 5. テスト計画

既存テスト資産を活かし、以下を追加する。

- unit
  - `tests/test_arm_history.py`
  - `tests/test_arm_selector.py`
  - `tests/test_arm_triple_acquirer.py`（小さなtoy triplesで）
  - `tests/test_witness_conflict_evaluator.py`
- integration
  - 既存の `test_pipeline_*` に倣い、armパイプラインの最小実行（1-2 iteration）を検証

## 6. 成果物（Deliverables）

- 実装
  - 新規モジュール（arm selector/history/pipeline/acquirer/evaluator）
  - 実行スクリプト（実験ディレクトリ入力で反復を回す）
- 記録
  - 反復ログ（arm_history, accepted_triples, diagnostics）
  - レポートMarkdown（最低限の集計でOK）

## 7. 未確定事項（着手前に決めたい）

1. 反復で「追加するトリプル」の定義
  - 合意済み: 主対象は evidence（body側）トリプル。
  - 残り: hypothesis（$r_*$）を生成する場合の扱い（保存先、LLM判定の入出力、KGへの反映タイミング）。

2. 反復中のルールプール
   - 原則固定（今回の前提）で進める。
   - 将来的に更新する場合のインターフェースをどう切るか。

3. 出力ディレクトリの形式
   - 既存の `experiments/<date>/iter-k/` に合わせるか
   - あるいは `IterationDataManager` を必須にするか

---

関連資料:
- combo-bandit overview: [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- v3 overview: [docs/rules/RULE-20260111-KG_REFINE_V3_OVERVIEW-001.md](../rules/RULE-20260111-KG_REFINE_V3_OVERVIEW-001.md)
- pipeline overview: [docs/rules/RULE-20260111-PIPELINE_OVERVIEW-001.md](../rules/RULE-20260111-PIPELINE_OVERVIEW-001.md)
- pipeline concrete: [docs/rules/RULE-20260111-PIPELINE_CONCRETE-001.md](../rules/RULE-20260111-PIPELINE_CONCRETE-001.md)
- 既存v3メイン: [main.py](../../main.py)
