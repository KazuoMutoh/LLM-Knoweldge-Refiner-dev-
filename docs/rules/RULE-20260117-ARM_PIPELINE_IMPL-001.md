# RULE-20260117-ARM_PIPELINE_IMPL-001: simple_active_refine/arm_pipeline.py 実装仕様（ArmDrivenKGRefinementPipeline）

作成日: 2026-01-17
最終更新日: 2026-02-01
更新日: 2026-01-17（新規エンティティ導入時に train_removed 由来の incident triples を追加する挙動を追記）
更新日: 2026-01-17（arm選択の説明可能性を高めるため、target/armの意味情報、evidence/witnessの永続化とselected_arms出力拡張を追記）
更新日: 2026-01-18（relation priors による witness 重み付け（raw保持）を追記）
更新日: 2026-01-22（incident triples の無効化/上限設定と、target score 重視・KG-FIT運用の注意点を追記）
更新日: 2026-01-24（candidate_source=local|web を導入し、Web取得候補による evidence 探索と provenance 出力、ターゲットサンプリング共有を追記）
更新日: 2026-02-01（web_entities.json 出力、Web entity ID 正規化・entity linking・web_provenance の保存形式を実装に合わせて明確化）
更新日: 2026-02-01（REC-20260128の検討を反映し、selectorへ entity2text を渡してトリプル例を自然言語化できるようにする接続契約を明確化）

---

## 0. 目的とスコープ

`simple_active_refine/arm_pipeline.py` は、arm（ルールの組）を「腕」として扱い、
**arm選択 → evidence取得 → proxy評価 → KG更新（evidence-only）**を反復実行し、
各イテレーションの成果物を `iter_k/` へ永続化するパイプライン実装を提供する。

本仕様は、現行実装（2026-01-17時点）の `ArmDrivenKGRefinementPipeline` / `ArmPipelineConfig` の
入力、出力、反復ごとの保存ファイル、選択・取得・評価の接続契約を再実装可能な粒度で定義する。

---

## 1. 提供API

### 1.1 データクラス: `ArmPipelineConfig`

必須:
- `base_output_path: str`（出力先）

主要パラメータ（デフォルト）:
- `n_iter: int = 1`
- `k_sel: int = 3`（各iterで選ぶarm数）
- `n_targets_per_arm: int = 50`（各armでサンプルするターゲット数）
- `max_witness_per_head: Optional[int] = None`

selector:
- `selector_strategy: str = "ucb"`（`ucb|epsilon_greedy|llm_policy|random` を想定）
- `selector_exploration_param: float = 1.0`（UCB）
- `selector_epsilon: float = 0.1`（ε-greedy）

proxy reward:
- `witness_weight: float = 1.0`
- `evidence_weight: float = 1.0`

candidate source（evidence探索用の候補集合）:
- `candidate_source: str = "local"`（`local|web`）
  - `local`: `candidate_triples_path`（典型: train_removed.txt）を候補集合として使用
  - `web`: 各iterで `LLMKnowledgeRetriever` により Web から候補トリプルを取得し、それを候補集合として使用

web retrieval（candidate_source=web のときのみ意味を持つ）:
- `web_llm_model: str`（例: gpt-4o）
- `web_use_web_search: bool`（OpenAI web_search_preview を使うか）
- `web_max_targets_total_per_iteration: int`（各iterの Web クエリ総数の上限）
- `web_max_triples_per_iteration: int`（各iterで保持する Web 候補トリプル数の上限）

relation priors（任意）:
- `relation_priors_path: Optional[str] = None`
  - 指定がある場合は、そのJSONを読み込んで witness を重み付けする
  - 指定がない場合でも、`<dir_triples>/relation_priors.json` が存在すれば自動検出して使用する
  - フォーマットは `predicate -> prior` または `{"meta":..., "payload":...}` を受理（payloadから `X` を抽出）

### 1.2 クラス: `ArmDrivenKGRefinementPipeline`

#### 1.2.1 コンストラクタ

- 入力:
  - `config: ArmPipelineConfig`
  - `arm_pool: list[ArmWithId]`
  - `rule_pool: AmieRules`
  - `kg_train_triples: list[Triple]`（現KGの初期）
  - `target_triples: list[Triple]`（対象トリプル集合）
  - `candidate_triples: list[Triple]`（候補集合。witness/evidence探索に使用）

備考:
- `candidate_source=web` の場合、`candidate_triples` は「各iterationで Web から取得した候補集合」に置き換わる。

内部状態（主要）:
- `kg_set: set[Triple]`（現KG。更新対象）
- `history: ArmHistory`（armごとの評価履歴）
- `selector: ArmSelector`（create_arm_selector で生成）
- `acquirer: LocalArmTripleAcquirer`（local候補集合から取得）
- `evaluator: WitnessConflictArmEvaluator`（proxy評価）
- `rule_by_key: dict[str, AmieRule]`（`str(rule)` をキーに逆引き）

#### 1.2.2 `from_paths(...)`（ファクトリ）

入力:
- `initial_arms_path`: `initial_arms.json` または `initial_arms.pkl`
- `rule_pool_pkl`: `initial_rule_pool.pkl`（AmieRules）
- `dir_triples`: データセットディレクトリ（`train.txt` を読む）
- `target_triples_path`: target triples TSV
- `candidate_triples_path`: candidate triples TSV

ロード規約:
- `load_arm_pool_with_ids(initial_arms_path)`
- `AmieRules.from_pickle(rule_pool_pkl)`
- `load_triples_tsv(dir_triples/train.txt)`
- `load_triples_tsv(target_triples_path)`
- `load_triples_tsv(candidate_triples_path)`

追加（任意）:
- `dir_triples/entity2textlong.txt`（優先）または `dir_triples/entity2text.txt` が存在する場合、TSV（`entity<TAB>text`）としてロードし `entity_texts: dict[str,str]` を構築する。
- `dir_triples/relation2text.txt` が存在する場合、TSV（`predicate<TAB>text`）としてロードし `relation_texts: dict[str,str]` を構築する（既存）。

#### 1.2.3 `run() -> None`

- `iteration=1..n_iter` を順に実行
- 各iterで `iter_dir = get_iteration_dir(base_output_path, iteration)` を作成
- 主要成果物を `iter_dir/` に保存

---

## 2. 反復処理フロー（各iteration）

1. arm選択
   - `selected_arms, policy_text = selector.select_arms(arm_pool, k=k_sel, iteration=iteration)`
   - selector生成:
     - `strategy=config.selector_strategy`
     - `ucb` の場合 `exploration_param`
     - `epsilon_greedy` の場合 `epsilon`

   1.1 付加情報（LLM-policy向けの文脈付与）
   - `dir_triples/relation2text.txt` が存在する場合、ターゲット関係やルールbody/headの関係を自然言語説明へマップして selector へ渡す。
   - `dir_triples/entity2textlong.txt` または `dir_triples/entity2text.txt` が存在する場合、entity の説明テキストをロードし selector へ渡す。
     - 目的: selector（LLM-policy）が target/evidence/added triples の例を「意味が読める」形に自然言語化して提示できるようにする（REC-20260128）。
   - `rule_pool` を参照し、各armに含まれるルールの
     - `body_predicates`
     - `head_predicates`
     - それぞれの説明文（relation2textがある場合）
     を `arm.metadata` に付与する（LLM-policyプロンプトでarmの「意味」を提示するため）。

2. 探索用インデックス構築
   - `candidate_source=local`:
     - `TripleIndex(list(kg_set) + candidate_triples)`
     - 「現KG + train_removed等の候補集合」を witness/evidence 探索空間にする
   - `candidate_source=web`:
     - 事前に Web から候補トリプルを取得し、`web_candidates` を構築する
     - `TripleIndex(list(kg_set) + web_candidates)`
     - 「現KG + Web候補集合」を witness/evidence 探索空間にする

3. evidence取得（acquire）
  - `LocalArmTripleAcquirer.acquire(...)`
  - 重要: ターゲットサンプリングはパイプライン側で決定し、acquirerへ渡して共有する（local/web で一致させるため）
    - seed固定（`random_seed=0` と iteration で派生）
  - 各ターゲットについて、arm内の各ルールを適用し witness数を数え、body triples（evidence）を収集

   3.1 relation priors による witness 重み付け（任意）
   - raw witness（整数）: `witness_by_arm_and_target`（置換数の合計）として保存
   - weighted witness（実数）: `witness_score_by_arm_and_target` として保存
     - ルール単位の重み $W(h)$ を body predicates の prior の積として定義する:
       $$
       W(h)=\prod_{p\in\mathrm{body\_predicates}(h)} X_p
       $$
     - ターゲット $t$ の weighted witness は $\sum_{h\in a} W(h)\,c_h(t)$ で集計する
     - prior 未定義の predicate は `default_relation_prior=1.0` を用いる

4. proxy評価（evaluate）
   - `WitnessConflictArmEvaluator.evaluate(acquisition, current_kg_triples=kg_set)`
   - v1ポリシー:
     - KGへ確定追加するのは **accepted_evidence_triples** のみ
     - ターゲット関係（hypothesis）は **pending_hypothesis_triples** に退避（store-only）

   4.1 追加の拡張（新規エンティティの incident triples 追加）
   - `accepted_evidence_triples` が **現KGに存在しないエンティティ**（subject/object）を導入した場合、
     その新規エンティティを含むトリプルを候補集合から追加で採用する。
     - `candidate_source=local`: `candidate_triples`（例: train_removed.txt）
     - `candidate_source=web`: `web_candidates`（Webから取得した候補集合）
   - これにより、「新規エンティティが evidence の1本だけで孤立しやすい」状況を緩和する。
   - 安全策として、ターゲット関係（hypothesis predicates）は incident 追加から除外する。
   - 本拡張は **設定で無効化/上限制御が可能**である（実装: `ArmPipelineConfig`）。
     - `add_incident_candidate_triples_for_new_entities`（既定: True）
     - `max_incident_candidate_triples_per_iteration`（既定: None。Noneのとき上限なし）
   - 運用メモ（target score 重視）:
     - incident triples は追加数が増えやすく、target score や Hits@k に対しノイズ源にもなり得る。
     - 特に **KG-FIT のようにテキスト属性で新規エンティティをある程度アンカーできる**設定では、
       TransE で導入した「孤立回避」の必要性が下がる可能性があるため、
       まずは「incident なし（OFF）」と「上限K（例: 0/50/200）」で A/B する。
   - 成果物として以下を出力する:
     - `accepted_incident_triples.tsv`（incident 追加分）
     - `accepted_added_triples.tsv`（accepted_evidence ∪ accepted_incident の和集合）

   4.2 witness の取り扱い（raw/weighted）
   - evaluator は、`acquisition.witness_score_by_arm_and_target` が存在する場合は、それを witness 合計として reward に使用する
   - raw witness 合計は `diagnostics.witness_total_raw` として保存し、説明・分析用途で参照可能にする

5. 履歴更新（ArmHistory）
   - 選択された各armについて `ArmEvaluationRecord` を作成し `history.add_record(...)`
   - `added_triples` は「そのarmが収集したevidenceのうち、現KGに未存在のもの」
   - `reward` は evaluator の `reward_by_arm[arm_id]`

  5.1 永続化する追加情報（説明可能性のため）
  - `ArmEvaluationRecord` は次も保存する。
    - `evidence_triples`: 取得したevidenceトリプル（追加されたかどうかに関わらず）
    - `witness_by_target`: ターゲットトリプルごとのwitness数（取得時点で観測された説明の厚み）
  - 目的: 「armが何を根拠に、どのターゲットをどれだけ説明したか」を後段（LLM-policy）で参照可能にする。

6. KG更新
   - evidence-only（従来）:
     - `for t in accepted_evidence_triples: kg_set.add(t)`
   - 拡張あり（現行）:
     - `for t in accepted_added_triples: kg_set.add(t)`

7. 永続化（iter_dir）
   - `selected_arms.json`
   - `accepted_evidence_triples.tsv`
   - `pending_hypothesis_triples.tsv`
   - `arm_history.pkl` / `arm_history.json`（その時点までの履歴全体）
   - `diagnostics.json`

  7.1 `selected_arms.json` の拡張
  - `selected_arms` の各要素に、armの `diagnostics`（coverage/witness/evidence関連度など）を埋め込む。
  - 目的: 実験ログ単体で「なぜそのarmが選ばれた/良かったか」を追跡できるようにする。

---

## 3. 出力仕様（iter_dir配下）

`iter_k/`（k=1..n_iter）:

- `selected_arms.json`
  - `iteration`, `k_sel`, `selector_strategy`, `policy_text`
  - `selected_arms`: list
    - `arm_id`, `arm_type`, `rule_keys`, `metadata`, `reward`
- `accepted_evidence_triples.tsv`
  - KGへ追加されたevidence triples（unique, sorted）
- `accepted_incident_triples.tsv`
  - 新規エンティティ導入に伴い、candidate（例: train_removed）から追加された incident triples（unique, sorted）
- `accepted_added_triples.tsv`
  - 実際にKGへ追加されたトリプルの全集合（accepted_evidence ∪ accepted_incident）（unique, sorted）
- `pending_hypothesis_triples.tsv`
  - witness>0 で支持されたが KGへは追加しない target triples（unique, sorted）
- `arm_history.pkl`
- `arm_history.json`
- `diagnostics.json`
  - evaluator由来の診断（witness_total, witness_total_raw, accepted_evidence_total, conflict_countなど）

`candidate_source=web` の場合、追加で次を出力する:
- `web_retrieved_triples.tsv`
  - Web から取得した候補トリプル（dedup済み）。探索用インデックスに投入される候補集合。
- `web_provenance.json`
  - キー: `"s\tp\to"`（TSV 1行相当）
  - 値（最小）: `{"source":"web","iteration":int,"arm_id":str,"url":str}`
  - 注意: snippet 等は現行では保存しない（必要なら別途拡張）

- `web_entities.json`
  - Web から導入された（主に新規）entity 情報の辞書。
  - キー: `web:<sha1_16>` の stable ID（`label|source_url` から生成）
  - 値: `{"label","description_short","description","source","iteration","arm_id", ...}`
  - entity linking が有効かつ成功した場合は `linked_to`（既存KGの entity_id）が付与される

Web entity ID の正規化（candidate_source=web の内部仕様）:
- `LLMKnowledgeRetriever` が返す新規entity（例: `e1`, `e2`）は、(label, source_url) が揃う場合に stable ID `web:<sha1_16>` へ正規化する
- entity linking が有効な場合、stable ID は既存KGの entity_id へ置換され得る（`web_entities.json` の `linked_to` と整合）

ディレクトリ命名:
- `get_iteration_dir(base, iteration)` は `iter_<iteration>` を返す（iterationは0以上を想定）

---

## 4. 失敗時の挙動

- ルールキーが `rule_by_key` に存在しない場合: `KeyError`（acquirer）
- 入力ファイルの読み込み失敗: 例外伝播
- その他: 例外伝播

---

## 5. 実装参照（ソースコード）

- 実装: [simple_active_refine/arm_pipeline.py](../../simple_active_refine/arm_pipeline.py)
- arm I/O: [simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py)
- selector: [simple_active_refine/arm_selector.py](../../simple_active_refine/arm_selector.py)
- acquirer: [simple_active_refine/arm_triple_acquirer_impl.py](../../simple_active_refine/arm_triple_acquirer_impl.py)
- evaluator: [simple_active_refine/arm_triple_evaluator_impl.py](../../simple_active_refine/arm_triple_evaluator_impl.py)
- iteration dir: [simple_active_refine/io_utils.py](../../simple_active_refine/io_utils.py)

---

## 6. 参考文献（docs）

- arm設計（witness/衝突、store-only、after評価導線）: [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- 統合ランナー仕様（このクラスの from_paths/run が呼ばれる）: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- 統合ランナー設計記録: [docs/records/REC-20260114-ARM_PIPELINE-001.md](../records/REC-20260114-ARM_PIPELINE-001.md)
- relation priors（witness重み付けの定義・運用）: [docs/rules/RULE-20260118-RELATION_PRIORS-001.md](RULE-20260118-RELATION_PRIORS-001.md)
