# RULE-20260117-ARM_SELECTOR_IMPL-001: simple_active_refine/arm_selector.py 実装仕様（ArmSelector / arm選択戦略）

作成日: 2026-01-17
最終更新日: 2026-02-01
更新日: 2026-02-01（REC-20260128の検討を反映し、entity2text の受け口・promptの意味展開/制約を仕様として明確化）

---

## 0. 目的とスコープ

`simple_active_refine/arm_selector.py` は、arm（=ルール組み合わせ）プールから **k 個のarmを選択**する戦略を提供する。
本仕様は、現行実装（2026-01-17時点）の arm selector の I/F・挙動・プロンプト規約を、再実装可能な粒度で定義する。

スコープ:
- selector の共通I/F（`ArmSelector.select_arms`）
- 実装済みの戦略（`llm_policy|ucb|epsilon_greedy|random`）
- LLM-policy における入力情報（arm統計・証拠トリプル・意味情報）と出力（policy_text の更新）

非スコープ:
- arm pool の生成（`build_initial_arms.py`）
- acquire/evaluate の詳細（witness定義や衝突判定）
- rule_pool の生成（AMIE+統合）

---

## 1. 提供API

### 1.1 データモデル

#### `ArmCandidate`
- 役割: `ArmWithId` から `arm_id` と `Arm` を保持する薄いラッパ。
- 主要メソッド:
  - `from_arm_with_id(arm_with_id: ArmWithId) -> ArmCandidate`

#### `ArmSelectionPolicy`（LLM structured output schema）
LLM-policy 戦略で、LLMから得る構造化出力のスキーマ。
- `reasoning: str`
- `policy_text: str`（次回以降へ持ち越す自然言語ポリシー）
- `selected_arm_ids: list[str]`
- `rationale_per_arm: dict[str, str]`

### 1.2 抽象基底: `ArmSelector`

#### コンストラクタ
- 入力:
  - `history: Optional[ArmHistory]`（省略時は空の `ArmHistory()`）
  - `target_predicates: Optional[list[str]]`（ターゲット関係の候補。LLM-policyで文脈提示に使用）
  - `relation_texts: Optional[dict[str,str]]`（predicate -> 説明文。`relation2text.txt` を想定）
  - `entity_texts: Optional[dict[str,str]]`（entity -> 説明文。`entity2textlong.txt` / `entity2text.txt` を想定）

#### 共通メソッド: `select_arms(...)`
- シグネチャ:
  - `select_arms(arm_pool: list[ArmWithId], k: int = 3, iteration: int = 0) -> tuple[list[ArmWithId], Optional[str]]`
- 出力:
  - `selected_arms`: 選ばれたarm（`ArmWithId`）
  - `policy_text`: LLM-policyのみ返しうる（それ以外は基本 `None`）

契約:
- `len(arm_pool) <= k` の場合は全armを返す。
- selector は arm pool を破壊的に変更しない。

---

## 2. 戦略（実装）

### 2.1 `LLMPolicyArmSelector`（`strategy="llm_policy"`）

#### 2.1.1 依存関係・LLM設定
- LangChain `ChatOpenAI` を使用する。
- デフォルト:
  - `chat_model="gpt-4o"`
  - `temperature=0.3`
- `OPENAI_API_KEY` は環境変数 `OPENAI_API_KEY` に設定される（`settings.OPENAI_API_KEY` をフォールバックに使用）。

#### 2.1.2 反復0の特例
- `iteration == 0` の場合、**arm_pool の先頭 k 個**を選ぶ（LLMは呼ばない）。
- `current_policy` が未設定なら、初期ポリシー文を設定して返す。

#### 2.1.3 LLMに提示する情報（prompt生成）

LLM-policyは「平均報酬」のような集計に加え、armが実際に target を説明していたかを判断できる情報を与える。

(1) ターゲット関係の文脈（Target Relation Context）
- `target_predicates`（最大5）を提示する。
- `relation_texts` がある場合は `predicate: description` 形式で提示する。
- `target_predicates` が空の場合は、`history.records` の末尾から `target_triples` のpredicateを推定する。

(2) Arm pool 統計（Current Arm Pool Statistics）
各armに対して次を提示する。
- arm識別: `arm_id`, `arm_type`, `rule_keys`
- 意味情報（任意）:
  - `arm.metadata["body_predicates"]` がある場合に列挙
  - `arm.metadata["body_predicate_texts"]` がある場合に説明文（最大5）をサンプル提示

(3) 履歴統計（ArmHistory -> ArmStatistics）
`history.get_all_arm_statistics()` を用いる。
- `Trials`（試行回数）
- `Mean reward`
- `Std reward`
- `Recent performance (last 3)`
- `Total triples added`

(4) 直近イテレーションの説明可能性コンテキスト（Last iter）
`history.get_records_for_arm(arm_id)` の末尾レコード（あれば）から、次を提示する。
- `diagnostics` の要約（例）:
  - `targets_total`, `targets_with_witness`, `target_coverage`
  - `mean_witness_per_target`
  - `evidence_total`, `evidence_new`, `evidence_existing`
  - `evidence_new_overlap_rate_with_targets`
- `witness_by_target` を用いて「witnessが高い target_triple 上位（最大3）」を表示
- `added_triples` のサンプル（最大5）
- `evidence_triples`（取得したevidence全体）のサンプル（最大5）

(4b) entity2text / relation2text による自然言語化（追加）
- prompt 内の triple 例は、可能なら次の形式で「意味が読める」形に整形する。
  - `<head_text> (head_id) — <predicate> [predicate_desc] — <tail_text> (tail_id)`
- `head_text` / `tail_text` は `entity2textlong.txt`（優先）または `entity2text.txt` をロードした `entity_texts` から取得する（無ければIDをそのまま使う）。
- `predicate_desc` は `relation2text.txt` をロードした `relation_texts` から取得する（無ければ省略）。
- prompt 長を抑えるため、entity/description は一定長で truncate し、triple サンプル数も小さく保つ。

(5) 現在の選択ポリシー（Current Selection Policy）
- `current_policy` を提示する（初回は未設定）。
- LLMには `policy_text` を「次回も再利用可能な基準」として更新させる。

(6) 指示（Your Task / Guidelines）
LLMに以下を明示する。
- 統計だけでなく armの意味（rule_keys / body predicates）と実際のevidenceを確認すること
- 「説明度」（coverage/witness）と「関連性」（target entity overlap 等）を評価すること
- 意味的整合性（semantic alignment）を考慮すること
  - 例: nationality なら地理/行政/居住など location 系 predicate が有益になりやすい

(6b) Semantic Grounding（追加: 意味を“強制”する）
- LLM に対して「意味的整合性を proxy reward と同等以上の重要度で扱う」ことを明示する。
- 各armを以下のルーブリックで評価するよう指示する（内部推論でよい）。
  - `SemanticAlignmentScore (0-2)`
  - `EvidenceRelevanceScore (0-2)`
  - `ProxyReliability (0-2)`
- 可能なら `SemanticAlignmentScore>=1` のarmを少なくとも1つ選ぶ、という制約を入れる（適合armが無い場合は理由を述べてfallback）。

##### 2.1.3.1 具体例（prompt抜粋）

前提:
- `k=2`, `iteration=5`
- `target_predicates=["/people/person/nationality"]`
- `relation_texts["/people/person/nationality"] = "nationality of a person"`（例）

このとき prompt は概ね次のような構造になる（説明のため簡略化・一部省略）。

```text
You are an expert in knowledge graph refinement using reinforcement learning principles.

Your task is to select 2 arms from the arm pool for iteration 5.

## Target Relation Context
The target predicate(s) (what we are trying to explain) are:
- /people/person/nationality: nationality of a person

## Current Arm Pool Statistics

### Arm 1: a1
**Type**: set
**Rule keys**: ['rk1', 'rk7']
**Body predicates (derived)**: ['/people/person/place_of_birth', '/location/location/containedby']
- **Body predicate descriptions (sample)**:
  - /people/person/place_of_birth: place where a person was born
  - /location/location/containedby: location containment
- **Trials**: 8
- **Mean reward**: 0.412300
- **Std reward**: 0.090000
- **Recent performance** (last 3): 0.380000
- **Total triples added**: 1200
- **Last iter**: 4
- **Target coverage**: 12 / 20  (rate=0.600)
- **Mean witness/target**: 1.250
- **Evidence acquired**: total=80, new=20, existing=60
- **New-evidence overlap w/ target entities**: 0.420
- **Top witnessed targets (last iter)**:
  - (p123 /people/person/nationality m456)  witness=3
  - (p777 /people/person/nationality m888)  witness=2
- **New evidence added (sample)**:
  - (p123 /people/person/place_of_birth loc9)
  - (loc9 /location/location/containedby locCountry)

### Arm 2: a2
... (省略)

## Current Selection Policy
(前回までの policy_text がここに入る。未設定なら “Start with exploration.”)

## Output Format
Provide a JSON with:
- reasoning
- policy_text
- selected_arm_ids
- rationale_per_arm
```

LLMの返却例（ArmSelectionPolicy）:

```json
{
  "reasoning": "a1 is semantically aligned with nationality via birthplace/location and shows decent coverage; a7 is undertried but has promising overlap.",
  "policy_text": "Prefer arms with location-related body predicates and non-trivial target coverage; keep 1 slot for undertried arms when coverage/overlap is unknown.",
  "selected_arm_ids": ["a1", "a7"],
  "rationale_per_arm": {
    "a1": "High coverage and location semantics (birthplace/containment).",
    "a7": "Exploration: low trials, potentially relevant semantics; verify in next iteration."
  }
}
```

#### 2.1.4 LLM出力の解釈とフェイルセーフ
- LLMの構造化出力 `ArmSelectionPolicy` を受け取る。
- `selected_arm_ids` を arm pool に照合し、存在するものだけを採用する。
- 指定数に満たない場合は、残りを `random.sample` で補完する。
- LLM呼び出し自体が例外の場合は、`random.sample(arm_pool, k)` にフォールバックする。

#### 2.1.5 返却値
- `select_arms` は `selected_arms[:k]` と `current_policy` を返す。

---

### 2.2 `UCBArmSelector`（`strategy="ucb"`）

- スコア:
  - $\mathrm{UCB}(a)=\mu(a)+c\sqrt{\ln(T+1)/n(a)}$
  - $\mu(a)$ は平均報酬、$n(a)$ は試行回数、$T$ は iteration（実装では引数 `iteration`）
- 未試行armはスコアを `inf` にして必ず探索される。
- `policy_text` は返さない（`None`）。

---

### 2.3 `EpsilonGreedyArmSelector`（`strategy="epsilon_greedy"`）

- 確率 $\varepsilon$ でランダム選択し、$1-\varepsilon$ で平均報酬最大を選ぶ。
- `k` 回の逐次選択で、同一armの重複選択を避ける。
- 未試行armの平均報酬は 0.0 として扱われる。
- `policy_text` は返さない（`None`）。

---

### 2.4 `RandomArmSelector`（`strategy="random"`）

- `random.sample(arm_pool, k)` を返す。
- `policy_text` は返さない（`None`）。

---

## 3. Factory: `create_arm_selector(...)`

- シグネチャ:
  - `create_arm_selector(strategy: str = "llm_policy", history: Optional[ArmHistory] = None, **kwargs) -> ArmSelector`
- 戦略名と実装クラスの対応:
  - `llm_policy` -> `LLMPolicyArmSelector`
  - `ucb` -> `UCBArmSelector`
  - `epsilon_greedy` -> `EpsilonGreedyArmSelector`
  - `random` -> `RandomArmSelector`
- 未知の戦略は `ValueError`。

`**kwargs` はパイプライン側から透過的に渡され、LLM-policyに限らず `target_predicates` / `relation_texts` を渡しても壊れない（受け口を全selectorが持つ）。

---

## 4. 入力データの前提（pipeline側との接続契約）

LLM-policyの「意味的整合性」や「説明度」を出すために、selectorが参照する情報は次から供給される。

- `ArmHistory`（`ArmEvaluationRecord`）:
  - `target_triples`, `added_triples`, `reward`, `diagnostics`
  - 追加コンテキスト: `evidence_triples`, `witness_by_target`
- `Arm.metadata`（pipelineが付与する可能性がある）:
  - `body_predicates: list[str]`
  - `head_predicates: list[str]`（LLM-policyの表示は現状 bodyのみだが、メタデータとしては付与されうる）
  - `body_predicate_texts: dict[str,str]`（relation2text由来）
  - `head_predicate_texts: dict[str,str]`（relation2text由来）
- `relation_texts: dict[str,str]`:
- `relation_texts: dict[str,str]`:
  - `dir_triples/relation2text.txt` をロードして構築されることを想定
- `entity_texts: dict[str,str]`:
  - `dir_triples/entity2textlong.txt`（優先）または `dir_triples/entity2text.txt` をロードして構築されることを想定

---

## 5. テスト方針

- LLM-policy は外部API依存のため、ユニットテストではLLM呼び出しをモックし、ネットワークに依存しないこと。
- 少なくとも以下を検証対象とする。
  - 反復0でLLMを呼ばず先頭k件を返す
  - `selected_arm_ids` に存在しないIDが含まれても安全にスキップし、k件に満たなければランダム補完する

---

## 6. 実装参照（ソースコード / 関連ドキュメント）

- 実装: [simple_active_refine/arm_selector.py](../../simple_active_refine/arm_selector.py)
- 履歴: [simple_active_refine/arm_history.py](../../simple_active_refine/arm_history.py)
- パイプライン（selectorへの入力付与）: [docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md](RULE-20260117-ARM_PIPELINE_IMPL-001.md)
- 設計記録（意図・背景）: [docs/records/REC-20260117-ARM_SELECTOR-001.md](../records/REC-20260117-ARM_SELECTOR-001.md)
- 設計記録（promptの意味展開/制約）: [docs/records/REC-20260128-LLM_SELECTOR_SEMANTIC_PROMPT-001.md](../records/REC-20260128-LLM_SELECTOR_SEMANTIC_PROMPT-001.md)
