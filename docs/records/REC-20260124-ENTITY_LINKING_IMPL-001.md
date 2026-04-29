# REC-20260124-ENTITY_LINKING_IMPL-001: Entity Linking機能の実装

作成日: 2026-01-24  
最終更新日: 2026-02-01

参照:
- 実験記録: [REC-20260124-NATIONALITY_WEB_TRIAL-001.md](./REC-20260124-NATIONALITY_WEB_TRIAL-001.md)
- 実装ファイル: [simple_active_refine/arm_pipeline.py](../../simple_active_refine/arm_pipeline.py)
- 実装ファイル: [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py)
- コアモジュール: [simple_active_refine/knoweldge_retriever.py](../../simple_active_refine/knoweldge_retriever.py) の `KnowledgeRefiner.find_same_entity`

---

## 0. 背景

Web取得実験（REC-20260124-NATIONALITY_WEB_TRIAL-001）において、以下の課題が判明：

### 問題
Web検索から取得したトリプルは新規entity（`web:<hash>`形式）を含むが、既存KGのentity（`/m/...`形式）とIDが異なるため、Hornルールのbody変数が統一（unify）できず、witnessが構成されない。

### 具体例
```
ルール: ?a /people/person/nationality ?b ← 
  ?b /location/location/contains ?f  (body atom 1)
  ?a /people/person/places_lived./people/place_lived/location ?f  (body atom 2)

Target triple: /m/01pcrw /people/person/nationality /m/09c7w0
  ?a = /m/01pcrw (既存entity)
  ?b = /m/09c7w0 (既存entity)

期待されるWeb取得トリプル（body atomsを満たすもの）:
1. /m/09c7w0 /location/location/contains web:619c3b695a5a954a  (?b → ?f)
2. /m/01pcrw /people/person/places_lived.../location web:619c3b695a5a954a  (?a → ?f)

問題: web:619c3b695a5a954a が既存KGに同じentity（例: /m/0xyz123）として存在する場合、
      IDが異なるため ?f が統一できず、witnessが成立しない
```

### 解決策
`KnowledgeRefiner.find_same_entity`を使用してWeb取得entityと既存KG entityを自動マッチングし、マッチした場合は既存IDに置換する。

例: `web:619c3b695a5a954a` (Nijlen) → `/m/0xyz123` (既存KGのNijlen)

```
1. /m/09c7w0 /location/location/contains /m/0xyz123  (?b → ?f)
2. /m/01pcrw /people/person/places_lived.../location /m/0xyz123  (?a → ?f)

→ ?f = /m/0xyz123 で統一でき、witnessが構成される
```

---

## 1. 実装概要

### 1.1 コア機能（既存）
`simple_active_refine/knoweldge_retriever.py`の`KnowledgeRefiner.find_same_entity`メソッド:
- 入力: Web取得entity（description付き）
- 処理:
  1. `TextAttributedKnoweldgeGraph.search_similar_entities`でベクトル・キーワード検索
  2. LLM（GPT-4o）で類似候補と同一性判定
- 出力: 同一と判定されたKG entity のリスト

### 1.2 統合実装（新規）
`simple_active_refine/arm_pipeline.py`の`_retrieve_web_candidates`メソッドに統合:

**変更箇所**:
1. `ArmPipelineConfig`に`web_enable_entity_linking: bool`フラグ追加（既定=True）
2. `ArmDrivenKGRefinementPipeline.__init__`に`kg: TextAttributedKnoweldgeGraph`引数追加
3. `from_paths`でWeb候補ソース有効時に`TextAttributedKnoweldgeGraph`を初期化
4. `_retrieve_web_candidates`で:
   - `KnowledgeRefiner`を初期化（`web_enable_entity_linking=True`の場合）
   - Web取得entityごとに`find_same_entity`を実行
   - マッチ結果を`entity_link_map: Dict[str, str]`に記録
   - トリプル構築時に`entity_link_map`でIDを置換

**処理フロー**:
```python
# 1. Web取得entityに対してstable IDを生成
web_id = _stable_id(label, source_url)  # "web:619c3b695a5a954a"

# 2. Entity linkingを実行（有効化されている場合）
if refiner and web_id not in entity_link_map:
    same_entities = refiner.find_same_entity(entity, top_k=5, similarity_threshold=0.7)
    if same_entities:
        entity_link_map[web_id] = same_entities[0].id  # "/m/03gh4"

# 3. トリプル構築時にマッピングを適用
ss = entity_link_map.get(ss, ss)  # web:619c3b695a5a954a -> /m/03gh4
oo = entity_link_map.get(oo, oo)
```

---

## 2. CLIオプション

### run_full_arm_pipeline.py

**新規オプション**:
```bash
--disable_entity_linking
```
- 説明: Entity linkingを無効化（既定=有効）
- 用途: デバッグ・比較実験時にentity linkingを無効化

**使用例**:
```bash
# Entity linking有効（既定）
python3 /app/run_full_arm_pipeline.py \
  --candidate_source web \
  --web_max_targets_total_per_iteration 5 \
  --web_max_triples_per_iteration 200 \
  # ... その他のオプション

# Entity linking無効
python3 /app/run_full_arm_pipeline.py \
  --candidate_source web \
  --disable_entity_linking \
  # ... その他のオプション
```

---

## 3. 動作確認

### テストスクリプト
`/app/tmp/debug/test_entity_linking_integration.py`

### テスト結果（2026-01-24）
```
Test 1 (Initialization): ✅ PASS
  - TextAttributedKnoweldgeGraph 初期化成功
  - KnowledgeRefiner 初期化成功

Test 2 (find_same_entity): ✅ PASS
  - Query: "Hawaii" (web:test123)
  - Result: /m/03gh4 にマッチング
  - Confidence: 1.00

Test 3 (ArmPipelineConfig): ✅ PASS
  - web_enable_entity_linking = True（既定値）
```

**重要な確認事項**:
- Web取得entity（"Hawaii"）が既存KG entity（`/m/03gh4`）と正しくマッチング
- LLMベースの同一性判定が機能（confidence=1.00）
- 類似だが異なるentity（`/m/02hrh0_`, `/m/014wxc`）は除外

---

## 4. 設計判断

### 4.1 Entity Linkingのタイミング
**選択**: Web取得直後、トリプル構築前に実行

**理由**:
- トリプル構築時に既に正しいIDを使用できる
- witnessマッチング時の複雑性を回避
- provenance保持が容易（元のweb IDを記録可能）

### 4.2 マッチング閾値
**既定値**:
- `top_k=5`: 類似候補を5件取得
- `similarity_threshold=0.7`: コサイン類似度0.7以上

**根拠**:
- テキストベースの類似検索では0.7が妥当（精度・再現率バランス）
- LLMによる最終判定があるため、候補は多めに取得

### 4.3 LLM判定の活用
**理由**:
- ベクトル類似度だけでは同一性を誤判定する可能性
- LLMがdescriptionの意味的整合性を判定
- Confidenceスコアで確信度を評価可能

### 4.4 既定値を「有効」に設定
**理由**:
- Web取得の主目的はKG拡張であり、entity linkingは必須機能
- 無効化は特殊用途（デバッグ・比較実験）のみ

---

## 5. パフォーマンス考慮事項

### 5.1 計算コスト
- **ベクトル検索**: O(n log n)（ChromaDB HNSW）
- **LLM判定**: O(top_k) × LLM API コール
- **全体**: entityあたり 2-5秒程度（top_k=5の場合）

### 5.2 最適化
- `entity_link_map`でキャッシュ（同一web IDの再計算を回避）
- バッチ処理は現状なし（将来検討）

### 5.3 スケーラビリティ
- **小規模実験**（5-10 entities/iteration）: 問題なし
- **大規模実験**（100+ entities/iteration）: LLM API制限に注意

---

## 6. 制限事項と既知の課題

### 6.1 新規entityの扱い
**制限**: KGに存在しないentityは`web:<hash>`のまま残る

**対応**:
- これは設計通りの動作（新規情報の追加が目的）
- 将来的にはWeb取得entityをKGに追加する機構が必要

### 6.2 曖昧なマッチング
**制限**: 複数の候補が同等に近い場合、最初の候補を選択

**対応**:
- LLMのconfidenceスコアでフィルタリング（閾値検討）
- 複数候補の扱い（将来検討）

### 6.3 LLM判定の不安定性
**制限**: LLMの出力が不安定な場合がある（レート制限、パース エラー等）

**対応**:
- リトライ機構（既に実装）
- エラー時はマッチングをスキップ（`web:<hash>`を保持）

---

## 7. 今後の拡張

### 7.1 バッチ処理
- 複数entityのentity linkingを並列化
- LLM API コールの削減

### 7.2 キャッシュ永続化
- `entity_link_map`をファイルに保存
- 次回実行時に再利用

### 7.3 マッチング品質の評価
- Precision/Recall計測
- Human-in-the-loopでの検証

---

## 結論（観測→含意→次アクション）

### 観測

- Web取得トリプルが `web:<hash>` を含むため、既存KG（`/m/...`）とIDが統一できず witness が成立しない問題を確認した。
- `KnowledgeRefiner.find_same_entity` を用い、Web entity を既存KG entity に自動マッチし、トリプル構築時にID置換する統合を実装した。
- `--disable_entity_linking` により、比較/デバッグ目的で無効化できる。

### 含意

- Web取得を evidence/witness に接続するには、entity linking（同一性判定）を「取得直後〜トリプル構築前」に挿入する設計が合理的で、後段のunify複雑化を避けられる。
- ただし LLM 判定コストが支配的になり得るため、iterationあたりのentity数上限とキャッシュ戦略が運用上の鍵になる。

### 次アクション

- linking結果（web_id→kg_id）の永続化と再利用で、APIコール数を抑制する。
- 閾値（similarity_threshold / confidence）をスイープし、precision/recall を計測して既定値を固める。

---

## 更新履歴
- 2026-01-24: 新規作成。Entity linking機能の実装記録。
