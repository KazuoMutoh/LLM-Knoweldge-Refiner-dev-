# REC-20260128-LLM_SELECTOR_SEMANTIC_PROMPT-001: LLM arm selector に意味情報を強制し、entity2text を用いた具体例を prompt に埋め込む計画

作成日: 2026-01-28
最終更新日: 2026-01-28

---

## 0. 背景 / 問題意識

現状の `LLMPolicyArmSelector` は、
- arm の統計（mean reward / recent performance / total added 等）
- last-iter diagnostics（coverage / witness / overlap）
- body predicates + relation2text（任意）
- last-iter の target/evidence/added triples のサンプル
を提示して、LLM に arm 選択と policy_text 更新を依頼している。

しかし、現状の prompt だと以下が起きやすい。

- 「意味（semantic alignment）」が *任意の助言* に留まり、LLM が “統計が良いarm” を選び続ける（proxy reward の最適化に吸い寄せられる）。
- トリプル例が `(s p o)` 形式で、entity の意味（人物/場所/組織など）が分からず、LLM が意味判断しにくい。
- target triple と evidence/added triples の “意味的にどう繋がるか” が prompt 中で説明されない。

ユーザ要望:
1) 「意味的な内容を踏まえて arm を選択すること」を prompt に追加（＝強制する）
2) target triple と arm が取得/追加した triple の具体例を、`entity2text` 等を使って prompt 内で説明

本記録は、まず実装計画（設計・変更点・検証方針）を定義する。

---

## 1. 現状把握（実装の入口）

- selector 実装: `simple_active_refine/arm_selector.py`
  - `LLMPolicyArmSelector._format_arm_statistics()` が、各armの統計 + last-iter の target/added/evidence トリプルを prompt に埋め込む。
  - 現状は entity の説明（entity2text）は selector に渡されていない。
- pipeline 側: `simple_active_refine/arm_pipeline.py`
  - `relation2text.txt` のロードは実装済み（`relation_texts`）。
  - `entity2text.txt` / `entity2textlong.txt` のロードは未実装。

既存資産:
- `simple_active_refine/knoweldge_retriever.py` は `entity2text(.long)` を読み込んで `entity_texts` を構築している（ただし web/entity-linking 用の経路）。
- 解析スクリプト群（例: `scripts/extract_combined_target_examples_with_context.py`）は `entity2text` を使った triple のテキスト化の実装例を持つ。

---

## 2. 目標（何を達成したら「できた」とするか）

### 2.1 Prompt 要件

- LLM に「意味的整合性を最優先の評価軸の一つとして扱う」ことを明示し、
  *統計だけで選ぶ行動* を抑制する（＝意味判断のフレームを prompt に組み込む）。

- prompt 内で、target triple と evidence/added triples の例を
  - entity2text（label/description）
  - relation2text（predicate の説明）
 で人間可読に展開する。

- LLM に「target triple を説明するために evidence がどう役立つか」を、
  *各armごとに短い自然言語の説明* として書かせる（rationale_per_arm と policy_text に反映させる）。

### 2.2 実装要件

- 既存の I/F を大きく壊さず、後方互換に拡張する。
- prompt のサイズが暴発しない（トークン超過を避ける）ための制御（サンプル数・文字数制限）を入れる。
- ユニットテストで「entity text を prompt に含めること」を最低限検証する。

---

## 3. 変更方針（実装計画）

### 3.1 データ注入: entity2text を selector に渡す

(1) `ArmSelector` に `entity_texts: Optional[Dict[str, str]]` を追加
- `__init__(..., entity_texts: Optional[Dict[str, str]] = None)`
- 内部保持: `self.entity_texts = dict(entity_texts) if entity_texts else {}`

(2) `ArmDrivenKGRefinementPipeline.from_paths()` で entity2text をロード
- `dir_triples/entity2textlong.txt` を優先し、なければ `entity2text.txt` を読む。
- TSV（entity<TAB>text）として読み込み、`Dict[str, str]` を生成。
- 既存の `relation2text` ロードと同様に “壊れている行はスキップ” を採用。

(3) `create_arm_selector(..., entity_texts=..., relation_texts=...)` で selector に渡す
- `llm_policy` 以外でも受け口を持つ（無視されてもよい）。

注: candidate_source=web の場合、`TextAttributedKnoweldgeGraph` が `entity_texts` を持つ。
- 初期段階では `dir_triples` の entity2text を基本ソースとし、将来的には `kg.entity_texts` とマージする（TODO）。

### 3.2 Prompt 生成: triple の「意味展開」を追加

(1) Triple 表示関数の拡張
- 現状: `(s p o)`
- 変更案: 
  - `head_text (head_id) — predicate (predicate_desc) — tail_text (tail_id)`
  - `head_text`/`tail_text` は `entity_texts.get(id, id)`
  - `predicate_desc` は `relation_texts.get(p)`

(2) 文字数制限（prompt サイズ制御）
- entity_text は最大 N 文字（例: 80）に truncate。
- triple サンプル数は現状より減らす/維持しつつ、テキスト化で情報密度を上げる。
  - Top witnessed targets: 3（現状維持）
  - Added evidence sample: 3（現状 5 → 3 へ）
  - Evidence acquired sample: 3（現状 5 → 3 へ）

(3) target/evidence の “関係づけ” を prompt に明示する
- 追加セクション案: `## Semantic Grounding`（固定文）
  - 「target predicate の意味」
  - 「このタスクで evidence が target を説明するとは何か」
  - 「semantic alignment の具体的判断基準（例: nationality ↔ location/birthplace/residence）」

### 3.3 「意味で選べ」を強制するプロンプト改修

現状は Guidelines に semantic alignment の一文があるだけなので、次を追加する。

(1) 明示的な採点ルーブリック
- LLM に、各armを以下の軸で短く評価させる（内部 reasoning でよい）。
  - `SemanticAlignmentScore`（0-2）: target predicate と body predicates / evidence が意味的に近いか
  - `EvidenceRelevanceScore`（0-2）: evidence が target entities に繋がっているか（overlap/例）
  - `ProxyScoreReliability`（0-2）: coverage/witness は良いが意味が薄い “水増し” っぽくないか

(2) 制約（ハード）
- `selected_arm_ids` のうち少なくとも1つは `SemanticAlignmentScore>=1` を満たすものを選べ、など。
  - ただし arm pool に適合が無い場合は例外を許容（fallback）。

(3) policy_text の更新ルール
- `policy_text` は「次回も使える具体基準」を必ず含める。
- 少なくとも以下を含める:
  - target predicate ごとの “好ましい predicate 群” の例
  - 高rewardでも意味が薄い arm を避ける条件（例: overlap低、target coverageのみ高い等）

### 3.4 仕様更新（rules）

- `docs/rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md`
  - 2.1.3 に、entity2text による triple 表示と、semantic grounding/ルーブリック/制約を追記。

---

## 4. 検証計画

### 4.1 ユニットテスト

- `tests/test_arm_selector.py` に以下を追加:
  - `LLMPolicyArmSelector(..., entity_texts={...}, relation_texts={...})` を構築
  - `_create_selection_prompt()` を直接呼び出して、
    - entity text（例: `"Barack Obama"`）
    - relation description（例: `"nationality of a person"`）
    が prompt に含まれることを assert
  - LLM呼び出しはモックのまま（既存方針維持）

### 4.2 実験（最小）

- 既存の nationality rerun 条件で `selector_strategy=llm_policy` を実行し、
  - selected_arms.json の rationale_per_arm / policy_text を確認
  - “意味に沿った根拠” が増えること
  - arm 選択が地理系 predicate を含む方向へ変化する兆しがあること
を観察する。

---

## 5. リスクと対策

- Prompt が長くなりすぎる
  - 対策: サンプル数削減、entity text の短縮、説明文テンプレの固定化。
- entity2text の品質が低い/欠損が多い
  - 対策: fallback を ID 表示にし、missing を明示。
- “意味を強制” が強すぎて探索が死ぬ
  - 対策: ルーブリックは優先度高だが、1枠は exploration に回す、などのルールを policy_text に含めさせる。

---

## 6. 次アクション（実装手順の粒度）

1) `simple_active_refine/arm_selector.py`
   - `ArmSelector` / `LLMPolicyArmSelector` に `entity_texts` を追加
   - triple のテキスト化関数（truncate含む）を追加
   - prompt を `Semantic Grounding` + ルーブリック/制約を含む形に更新
2) `simple_active_refine/arm_pipeline.py`
   - `from_paths()` に entity2text ロードを追加し selector に渡す
3) `tests/test_arm_selector.py`
   - prompt に entity/relation text が含まれることを検証するテストを追加
4) `docs/rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md`
   - prompt 仕様を更新（entity2text 表示・ルーブリック・制約）
5) `docs/database/index.md`
   - 本RECと更新した RULE を台帳に反映

---

## 7. 参照

- 実装: `simple_active_refine/arm_selector.py`
- pipeline: `simple_active_refine/arm_pipeline.py`
- rules: `docs/rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md`
- entity2text 読み込み例: `simple_active_refine/knoweldge_retriever.py` / `scripts/extract_combined_target_examples_with_context.py`
