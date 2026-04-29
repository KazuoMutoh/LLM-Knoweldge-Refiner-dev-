# RULE-20260124-WEB_ENTITY_RETRIEVAL-001: Webからのentity取得標準仕様

作成日: 2026-01-24
最終更新日: 2026-02-01
更新日: 2026-02-01（ArmDrivenKGRefinementPipeline の candidate_source=web 実装に合わせ、stable web ID / entity linking / 出力ファイル仕様を更新）

## 目的
本プロジェクトにおいて、LLMとWeb検索を活用して外部知識源からentityとtripleを取得する機能（`LLMKnowledgeRetriever`）の標準仕様を定義する。これにより、
- ローカル候補集合（train_removed）では得られない新規entityと関係の発見
- Hornルールのbodyパターンに基づく証拠トリプルの取得
- Web検索結果のURL provenance付き構造化データの生成
を標準化し、再現性と品質を担保する。

## スコープ
- 対象コード:
  - [simple_active_refine/knoweldge_retriever.py](../../simple_active_refine/knoweldge_retriever.py) の `LLMKnowledgeRetriever` クラス
  - [simple_active_refine/arm_pipeline.py](../../simple_active_refine/arm_pipeline.py) の `ArmDrivenKGRefinementPipeline._retrieve_web_candidates()`（candidate_source=web）
  - （補助）[simple_active_refine/triple_acquirer_impl.py](../../simple_active_refine/triple_acquirer_impl.py) の `WebSearchTripleAcquirer`（旧来の rule単位acquirer。現行arm-pipelineでは未使用）
- 対象機能:
  - `retrieve_knowledge()`: Hornルールのbodyパターンに基づく知識取得
  - `retrieve_knowledge_for_entity()`: 単一entityに関する関係/トリプル取得
- 対象外:
  - LLMによるHornルール生成（`BaseRuleGenerator`）
  - TextAttributedKnowledgeGraphの管理機能

## 背景
従来のKG改善アプローチでは、ローカル候補集合（train_removedなど）から証拠トリプルを選択してきた。しかし、これには以下の制約があった:
- 候補集合に存在しないentity/tripleは追加できない
- 新規entityの発見やOpen World前提の探索が困難

Web検索とLLMを組み合わせることで:
- Hornルールのbodyパターンを満たす新規entityを動的に発見
- 実世界の事実に基づく証拠トリプルを取得
- URL provenanceにより検証可能性を確保

参照（検討記録）:
- [docs/records/REC-20260123-ARM_WEB_RETRIEVAL-001.md](../records/REC-20260123-ARM_WEB_RETRIEVAL-001.md)
- [docs/records/REC-20260124-NATIONALITY_WEB_TRIAL-001.md](../records/REC-20260124-NATIONALITY_WEB_TRIAL-001.md)

参照（関連ルール）:
- [docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md](./RULE-20260117-ARM_PIPELINE_IMPL-001.md): candidate_source=web の統合

## LLMKnowledgeRetriever クラス

### 初期化パラメータ
```python
def __init__(
    self, 
    kg: TextAttributedKnoweldgeGraph = None,
    llm_model: str = 'gpt-4o', 
    use_web_search: bool = True
)
```

**パラメータ仕様**:
- `kg`: entity/relationのテキスト記述を参照するための知識グラフ（任意、Noneの場合はID文字列のみ使用）
- `llm_model`: 使用するLLMモデル名（デフォルト: `'gpt-4o'`）
- `use_web_search`: OpenAI Responses APIのweb_search_preview機能を使用するか（デフォルト: `True`）

**内部実装要件**:
- `use_web_search=True` の場合、`openai.OpenAI` クライアントを初期化し、`responses.create()` API を使用
- `use_web_search=False` の場合、LangChain経由で標準のLLM呼び出しを使用（Web検索なし）

### 主要メソッド: retrieve_knowledge()

#### 呼び出し仕様
```python
def retrieve_knowledge(
    self,
    target_triples: List[Tuple],
    rules: AmieRules
) -> List[RetrievedKnowledge]
```

**入力**:
- `target_triples`: 対象トリプルのリスト（各要素は `(subject, predicate, object)` タプル）
- `rules`: `AmieRules` オブジェクト（Hornルール集合）

**出力**:
- `List[RetrievedKnowledge]`: 各target tripleとruleの組み合わせに対する取得結果のリスト

**処理フロー**:
1. 各target tripleについて:
   - head変数（?a, ?b）とtarget tripleの対応づけ
   - kgからentity/relationのテキスト記述（label, description_short, description）を取得
2. 各ruleについて:
   - bodyパターン（例: `?a /people/person/place_of_birth ?c, ?c /location/location/contains ?b`）を文字列化
   - 既知変数（?a, ?b）のテキスト記述をプロンプトに含める
   - LLMにbodyパターンを満たす新規entity（?c, ?d等）の発見を依頼
3. LLM応答のパース:
   - JSON形式で返されたtriples/entitiesをパース
   - 既存entity IDはそのまま保持、新規entityには連番ID（e1, e2, ...）を付与
4. `RetrievedKnowledge(triples=..., entities=...)` を作成して返す

#### プロンプト設計標準

**必須要素**:
- Target tripleの説明（head変数のテキスト記述）
- Hornルールのbodyパターンの明示
- 変数マッピング（?a=既知entity, ?b=既知entity, ?c=新規発見対象）
- 構造化JSON出力の要求（厳密なスキーマ提示）

**出力JSON スキーマ**:
```json
{
    "triples": [
        {
            "subject": "entity_id (既存IDまたはe1, e2, ...)",
            "predicate": "relationship",
            "object": "entity_id",
            "source": "完全なURL (https://... で始まる)"
        }
    ],
    "entities": [
        {
            "id": "e1",
            "label": "Entity proper name (rdfs:labelに相当)",
            "description_short": "Entity proper name (labelと同じ値)",
            "description": "詳細説明（2-3文）またはnull",
            "source": "完全なURL"
        }
    ]
}
```

**プロンプト設計の注意事項**:
1. **具体的な指示**: "Find 1-3 triples that match the body patterns" のように具体的な数を指定
2. **プレースホルダー禁止**: "Avoid generic placeholders like [Person's Name], [City Name]" と明示
3. **ソース必須**: "Provide source information as COMPLETE URLs" と強調
4. **ターゲット除外**: "Do NOT create the target triple itself" と明記
5. **実名要求**: "Use REAL entity names and descriptions" を指示

#### エラーハンドリング

**必須処理**:
- JSON パースエラー時: 空の `RetrievedKnowledge(triples=[], entities=[])` を返す
- Web検索API エラー時: フォールバック（標準LLM）を試行
- 例外発生時: ログ出力（トレースバック付き）+ 空のRetrievedKnowledgeを返す

**ログレベル**:
- `logger.info()`: 処理開始、成功時（取得件数）
- `logger.warning()`: JSON不在、Web検索失敗（フォールバック）
- `logger.error()`: パースエラー、例外発生（トレースバック付き）
- `logger.debug()`: プロンプト内容、LLM応答（最初の300文字）

### 主要メソッド: retrieve_knowledge_for_entity()

#### 呼び出し仕様
```python
def retrieve_knowledge_for_entity(
    self,
    entity: Entity,
    list_relations: List[Relation]
) -> RetrievedKnowledge
```

**入力**:
- `entity`: 検索対象のEntityオブジェクト（id, label, description_short, description）
- `list_relations`: 候補となるRelationのリスト

**出力**:
- `RetrievedKnowledge`: 取得されたtriples/entities

**処理フロー**:
1. LLMにentityのdescriptionとrelationリストを提示し、関連するrelationを選択させる（1個以上）
   - 各relationについて、entityがhead（subject）またはtail（object）になり得るかを判定
2. 選択されたrelationごとに:
   - tripleパターン（例: `(entity, relation, ?)` または `(?, relation, entity)`）を構築
   - Web検索で?に該当するentityを発見
   - 新規entityに連番ID（e1, e2, ...）を付与
3. 取得したtriples/entitiesを集約してRetrievedKnowledgeを返す

**特記事項**:
- relation選択とtriple検索の2段階プロンプト
- position（head/tail）を明示的に指定
- relationのdescription（description_short + description）を活用

補足（directionality の運用）:
- `Relation.position` は retrieval の向きをヒントとして与える。
  - `head`: (entity, relation, ?)
  - `tail`: (?, relation, entity)
- `ArmDrivenKGRefinementPipeline` 側では、ルールbodyの「head変数に接続している向き」から (predicate, position) を推定して `Relation.position` に設定する。

## Entity ID の割り当てルール

### 既存entityと新規entityの区別

**既存entity**:
- target tripleで既知の変数（?a, ?b）に対応するentity
- knowledge graphに既に存在するentity
- ID形式: FB15k-237形式（`/m/xxxxx` など）やKG固有のID

**新規entity**:
- Hornルールのbodyパターンで新たに発見される変数（?c, ?d, ...）
- Web検索により取得したentity
- ID形式: 連番（`e1`, `e2`, `e3`, ...）

### ID割り当ての実装標準

**LLMへの指示**（プロンプトに含める）:
```
For existing entities (?a, ?b from target triple), use their actual IDs (e.g., /m/02mjmr, /m/09c7w0)
For NEW entities (?c, ?d, etc. discovered during search), assign sequential IDs: e1, e2, e3, etc.
```

**パース時の処理**:
- LLM応答のJSONをそのまま使用（IDの再割り当ては行わない）
- `label` と `description_short` はentityの proper name（実名）を使用
- 新規entityの `id` は `e1`, `e2`, ... の形式であることを想定

### ID衝突の回避

Web検索で取得した新規entityのIDは、既存KGのentity IDと衝突しないように:
- 連番ID（`e1`, `e2`, ...）は既存KGでは使用されない形式であることを前提
- もし衝突の可能性がある場合は、prefix付き（例: `web_e1`, `llm_e1`）を検討

### stable web ID（現行標準）

candidate_source=web（arm-pipeline統合）では、LLMが返す連番ID（`e1`, `e2`, ...）をそのままKGへ持ち込まず、次を標準とする:

- (label, source_url) が揃う場合:
  - stable ID `web:<sha1_16>` を生成し、同一の (label, source_url) は iteration を跨いでも同一IDに正規化する
- 上記が揃わない場合（フォールバック）:
  - `web:iter{iteration}:{arm_id}:{raw_id}` のような iteration スコープのIDを許容する（ただし再現性は弱い）

さらに entity linking が有効な場合:
- stable web ID は既存KGの entity_id に置換され得る（= 既存IDへ吸収）
- 置換結果は `web_entities.json` の `linked_to` に保存する

## Provenance（出典情報）の管理

### 必須要件

**すべてのtriple/entityに完全なURL sourceを付与**:
- 形式: `https://...` または `http://...` で始まる完全なURL
- 例: `https://en.wikipedia.org/wiki/Barack_Obama`
- 不完全URL（`wikipedia.org`, `en.wikipedia.org/...`）は不可

**プロンプトでの明示**:
```
Include COMPLETE URL sources (must start with http:// or https://)
Provide source information as COMPLETE URLs (e.g., https://en.wikipedia.org/wiki/Hawaii)
```

### 保存と活用

**出力データへの記録**:
- `Triple` オブジェクトの `source` フィールドに格納
- `Entity` オブジェクトの `source` フィールドに格納
- arm-run の `accepted_added_triples.jsonl` に provenance を含める

candidate_source=web の arm-run では、iteration成果物として provenance を別ファイルに保存する:
- `iter_k/web_provenance.json`
  - キー: `"s\tp\to"`
  - 値: `{"source":"web","iteration":k,"arm_id":...,"url":...}`

**検証可能性**:
- ユーザが手動でURLを確認可能
- 将来的な自動検証（URLクローリング、事実チェック）の基盤

## WebSearchTripleAcquirer との統合

### 役割分担

**LLMKnowledgeRetriever**:
- LLMとWeb検索のコア実装
- Hornルールに基づく知識取得ロジック
- JSON構造化出力のパース

**WebSearchTripleAcquirer**（`triple_acquirer_impl.py`）:
- `BaseTripleAcquirer` インターフェースの実装
- target tripleのサンプリング（`n_targets_per_rule`）
- 取得結果のTripleオブジェクトへの変換
- dump_base_dirへの結果保存（任意）

**LLMWebTripleAcquirer**（`pipeline_concrete.py`）:
- 軽量なwrapperクラス
- 依存性注入（retriever, retrieval_fn）のサポート
- テスト時のモック化を容易にする

### データフロー

```
target_triples + rules
  ↓
WebSearchTripleAcquirer.acquire()

---

## candidate_source=web（arm pipeline）での成果物標準

`ArmDrivenKGRefinementPipeline`（[simple_active_refine/arm_pipeline.py](../../simple_active_refine/arm_pipeline.py)）で `candidate_source=web` を指定した場合、各 iteration `iter_k/` に次を保存する。

- `web_retrieved_triples.tsv`
  - Webから取得した候補トリプル（dedup済み、hypothesis predicateは除外済み）
  - stable web ID / entity linking の結果が反映された subject/object を含み得る

- `web_provenance.json`
  - `"s\tp\to" -> {"source":"web","iteration":k,"arm_id":...,"url":...}`

- `web_entities.json`
  - `web:<sha1_16> -> {label, description_short, description, source, iteration, arm_id, linked_to?}`

品質管理（最低限）:
- hypothesis predicate（ターゲット関係）は Web候補集合へ混入させない（leak防止）
- `web_max_targets_total_per_iteration` / `web_max_triples_per_iteration` を用いて外部取得の上限を必ず設定する（コスト/ノイズ/再現性のため）
  ↓ (n_targets_per_rule でサンプリング)
LLMKnowledgeRetriever.retrieve_knowledge()
  ↓ (OpenAI responses.create with web_search_preview)
List[RetrievedKnowledge]
  ↓ (Triple/Entityオブジェクトへ変換)
TripleAcquisitionResult(candidates_by_rule={...})
```

## 品質管理とベストプラクティス

### プロンプト品質の確保

**定期的な見直し**:
- LLM応答の品質（実名使用、プレースホルダーなし）をモニタリング
- JSON パースエラー率の監視（10%未満を目標）
- ソースURL の完全性チェック（http/httpsで始まるか）

**バージョン管理**:
- プロンプトテンプレートの変更履歴を記録
- A/Bテストによる改善効果の検証

### パフォーマンス

**API コスト削減**:
- `n_targets_per_arm` パラメータでWeb検索呼び出し回数を制御
- キャッシュ機構の導入検討（同一triple+ruleの再取得を避ける）

**タイムアウト/リトライ**:
- OpenAI API のタイムアウト設定（デフォルト値を使用）
- リトライは OpenAI SDK のデフォルト実装に依存

### 結果の検証

**手動レビュー**:
- arm-run実行後、`accepted_added_triples.jsonl` をサンプリング
- sourceフィールドのURL を実際に確認
- entity label の実名確認（プレースホルダーでないか）

**自動チェック**（推奨）:
- URL形式の正規表現検証
- label/description_short が `[...]` パターンを含まないことを確認
- 新規entity IDが `e1`, `e2`, ... 形式であることを確認

## 制限事項と将来の拡張

### 現在の制限

**LLMの制約**:
- Web検索結果の品質はOpenAI APIに依存
- 誤情報や不正確な事実が含まれる可能性

**スケーラビリティ**:
- 大量のtarget tripleに対する並列化は未実装（順次処理）
- API rate limitへの対応は呼び出し側で制御

**entity名寄せ**:
- 同一entityの異なる表記（例: "USA" vs "United States"）の統合は未実装

### 将来の拡張案

**検証機能の強化**:
- URL crawlingによる事実検証
- 複数ソース間の整合性チェック
- entity名寄せ（entity resolution）

**検索戦略の改善**:
- Web検索以外の情報源（DBpedia、Wikidata API）の統合
- マルチステップ推論（chain-of-thought）の導入

**並列化**:
- 複数target tripleの並列処理（async/await）
- バッチAPIの活用

## 関連ドキュメント

**設計ルール**:
- [RULE-20260117-ARM_PIPELINE_IMPL-001.md](./RULE-20260117-ARM_PIPELINE_IMPL-001.md): candidate_source=web の統合仕様
- [RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](./RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md): arm-driven refinement の全体像

**検討記録**:
- [REC-20260123-ARM_WEB_RETRIEVAL-001.md](../records/REC-20260123-ARM_WEB_RETRIEVAL-001.md): Web取得統合の実装計画
- [REC-20260124-NATIONALITY_WEB_TRIAL-001.md](../records/REC-20260124-NATIONALITY_WEB_TRIAL-001.md): Web候補取得の観察実験

**外部情報**:
- OpenAI Responses API: [docs/external/](../external/)（該当ドキュメントがある場合）

---
最終更新: 2026-01-24
