# REC-20260124-WEB_ENTITY_PERSISTENCE-001: Web取得Entity情報の永続化実装

**DOC_ID**: REC-20260124-WEB_ENTITY_PERSISTENCE-001  
**作成日**: 2026-01-24  
**最終更新日**: 2026-02-01（更新: 結論の統一フォーマット化）  
**関連ドキュメント**:
- [REC-20260124-ENTITY_LINKING_IMPL-001](./REC-20260124-ENTITY_LINKING_IMPL-001.md) - Entity Linking実装記録
- [REC-20260124-NATIONALITY_WEB_TRIAL-001](./REC-20260124-NATIONALITY_WEB_TRIAL-001.md) - Nationality Web取得実験記録
- [REC-20260123-ARM_WEB_RETRIEVAL-001](./REC-20260123-ARM_WEB_RETRIEVAL-001.md) - ARM Web取得機能設計

---

## 概要

Web取得によって得られたentity情報（label, description_short, description, source）を`web_entities.json`として永続化する機能を実装した。さらに、Web取得されたentity情報を`TextAttributedKnoweldgeGraph`に統合し、次のiterationでの学習に反映できるようにした。これにより、Web取得で生成されたTAKG（Text-Attributed Knowledge Graph）のentity属性情報が保存され、継続的に活用されるようになった。

---

## 背景

### 問題

- Web取得実験（`REC-20260124-NATIONALITY_WEB_TRIAL-001`）で24個のトリプルが取得されたが、entity情報（description等）がファイルに保存されていなかった
- `web_provenance.json`にはトリプル単位の出典情報（URL）のみが保存され、entity自体のテキスト属性情報は失われていた
- `LLMKnowledgeRetriever.retrieve_knowledge_for_entity()`が返す`RetrievedKnowledge`にはentity情報（`Entity`オブジェクト）が含まれているが、`arm_pipeline.py`の`_retrieve_web_candidates()`メソッドでトリプル追加処理のみに使用され、entity情報自体は保存されていなかった

### 影響

- Web取得されたentityのdescription（TAKGの重要な要素）を後から確認できない
- Entity linking結果の検証が困難（どのweb:hash IDがどのようなdescriptionを持つentityだったかが不明）
- デバッグや分析時にLLMがどのような情報を取得したかを追跡できない

---

## 実装内容

### 修正ファイル

**`/app/simple_active_refine/arm_pipeline.py`**

### 変更点

#### 1. `_retrieve_web_candidates()`メソッドの戻り値を拡張

**変更前**:
```python
def _retrieve_web_candidates(
    self,
    *,
    selected_arms: List[ArmWithId],
    targets_by_arm: Dict[str, List[Triple]],
    iteration: int,
) -> tuple[List[Triple], Dict[str, Dict]]:
```

**変更後**:
```python
def _retrieve_web_candidates(
    self,
    *,
    selected_arms: List[ArmWithId],
    targets_by_arm: Dict[str, List[Triple]],
    iteration: int,
) -> tuple[List[Triple], Dict[str, Dict], Dict[str, Dict]]:
```

第3戻り値として`web_entities_dict: Dict[str, Dict]`を追加。

#### 2. メソッド内に`web_entities_dict`を追加

```python
out_triples: List[Triple] = []
provenance: Dict[str, Dict] = {}
seen: Set[Triple] = set()
web_entities_dict: Dict[str, Dict] = {}  # web_entity_id -> entity info
entity_link_map: Dict[str, str] = {}  # web_entity_id -> kg_entity_id
```

#### 3. Entity情報の保存処理を追加

Web IDを生成した直後に、entity情報をweb_entities_dictに格納：

```python
# Store entity information in web_entities_dict
if web_id not in web_entities_dict:
    web_entities_dict[web_id] = {
        "label": label,
        "description_short": str(getattr(e, "description_short", "") or ""),
        "description": str(getattr(e, "description", "") or ""),
        "source": url,
        "iteration": int(iteration),
        "arm_id": arm_id
    }
```

#### 4. Entity Linkingの結果も記録

Entity linkingが成功した場合、紐付け先のKG entity IDを`linked_to`フィールドに記録：

```python
if same_entities:
    matched_entity = same_entities[0]
    entity_link_map[web_id] = matched_entity.id
    web_entities_dict[web_id]["linked_to"] = matched_entity.id
    logger.info("[web] Entity linking: %s -> %s (%s)", web_id, matched_entity.id, label)
```

#### 5. 呼び出し元で`web_entities.json`として保存

```python
web_candidates: List[Triple] = []
web_provenance: Dict[str, Dict] = {}
web_entities: Dict[str, Dict] = {}
if self.config.candidate_source == "web":
    web_candidates, web_provenance, web_entities = self._retrieve_web_candidates(
        selected_arms=selected_arms,
        targets_by_arm=targets_by_arm,
        iteration=iteration,
    )
    # Persist raw web candidates + provenance + entities for audit/repro.
    write_triples(iter_dir / "web_retrieved_triples.tsv", web_candidates)
    save_json(iter_dir / "web_provenance.json", web_provenance)
    save_json(iter_dir / "web_entities.json", web_entities)
```

#### 6. Web取得entityの`TextAttributedKnoweldgeGraph`への追加（新規）

Web取得されたentity情報を`self.kg`（`TextAttributedKnoweldgeGraph`）に追加：

```python
# Add web-retrieved entities to KG (for next iteration's TAKG)
if self.kg and web_entities:
    from simple_active_refine.knoweldge_retriever import Entity as KR_Entity
    web_entity_list = []
    for entity_id, entity_info in web_entities.items():
        web_entity = KR_Entity(
            id=entity_id,
            label=entity_info.get("label", entity_id),
            description_short=entity_info.get("description_short", ""),
            description=entity_info.get("description", ""),
            source=entity_info.get("source", "")
        )
        web_entity_list.append(web_entity)
    self.kg.add_entities(web_entity_list)
    logger.info("[web] Added %d web entities to KG (TAKG)", len(web_entity_list))
```

これにより、Web取得されたentityのdescription情報が`self.kg.entity_texts`に追加され、次のiteration実行時にChromeDB（ベクトル検索）とBM25（キーワード検索）のインデックスにも反映される。

#### 7. 受け入れられたトリプルの`TextAttributedKnoweldgeGraph`への追加（新規）

`accepted_added_triples`を`self.kg`に追加：

```python
# Add accepted triples to TextAttributedKnoweldgeGraph (for next iteration's TAKG)
if self.kg and accepted_added_triples:
    from simple_active_refine.knoweldge_retriever import Triple as KR_Triple
    kg_triple_list = [
        KR_Triple(subject=s, predicate=p, object=o)
        for s, p, o in accepted_added_triples
    ]
    self.kg.add_triples(kg_triple_list, data_type='train')
    logger.info("[kg] Added %d accepted triples to KG (TAKG)", len(kg_triple_list))
```

#### 8. 各iterationの最後に`TextAttributedKnoweldgeGraph`をファイルに保存（新規）

```python
# Save TextAttributedKnoweldgeGraph to files (entity2text.txt, entity2textlong.txt, train.txt)
if self.kg:
    self.kg.save_to_files()
    logger.info("[kg] Saved TextAttributedKnoweldgeGraph to files for iteration %d", iteration)
```

これにより、Web取得されたentity情報が`entity2text.txt`と`entity2textlong.txt`に追加され、トリプルが`train.txt`に追加される。次のiterationでの埋め込みモデル学習時に、これらのファイルが読み込まれ、Web取得情報が学習に反映される。

---

## 成果物

### 新規ファイル

**`<run_dir>/arm_run/iter_<N>/web_entities.json`**

```json
{
  "web:619c3b695a5a954a": {
    "label": "Nijlen",
    "description_short": "Nijlen",
    "description": "Nijlen is a municipality in Belgium...",
    "source": "https://en.wikipedia.org/wiki/Nijlen",
    "iteration": 1,
    "arm_id": "arm_268fbd2e5a65",
    "linked_to": "/m/0123abc"  // Entity linkingが成功した場合のみ
  },
  ...
}
```

### フィールド説明

| フィールド | 型 | 説明 |
|-----------|---|------|
| `label` | str | Entityの名称（rdfs:labelに相当） |
| `description_short` | str | 短い説明文（labelと同じ値が多い） |
| `description` | str | 詳細な説明文（LLMがWebから取得した2-3文程度の説明） |
| `source` | str | 出典URL（Wikipedia等） |
| `iteration` | int | 取得されたiteration番号 |
| `arm_id` | str | 取得に使用されたarm ID |
| `linked_to` | str (optional) | Entity linkingで紐付けられた既存KG entity ID |

### TAKG統合ファイル（自動更新）

Web取得と受け入れ処理の後、各iterationの最後に以下のファイルが更新される：

**`<dataset_dir>/entity2text.txt`**: entity IDとdescription_shortのマッピング
```
/m/02mjmr	Barack Obama
web:619c3b695a5a954a	Nijlen
...
```

**`<dataset_dir>/entity2textlong.txt`**: entity IDと詳細descriptionのマッピング
```
/m/02mjmr	Barack Hussein Obama II is an American politician who served as the 44th president...
web:619c3b695a5a954a	Nijlen is a municipality located in the Belgian province of Antwerp...
...
```

**`<dataset_dir>/train.txt`**: トリプルデータ（TAB区切り）
```
/m/02mjmr	/people/person/nationality	/m/09c7w0
/m/02mjmr	/people/person/place_of_birth	web:619c3b695a5a954a
...
```

---

## 検証

### テスト実行

既存の実験ディレクトリで`web_entities.json`が存在することを確認：

```bash
find /app/experiments/20260124/exp_web_trial_nationality_iter1 -name "web_entities.json"
```

### 期待される動作

- `candidate_source="web"`時、各iterationで`web_entities.json`が生成される
- Web取得されたすべてのentityについて、テキスト属性情報が保存される
- Entity linkingが有効な場合、`linked_to`フィールドに既存entity IDが記録される

---

## 影響範囲

### 変更されたコンポーネント

- `ArmPipeline._retrieve_web_candidates()`: 戻り値の拡張（web_entities追加）
- `ArmPipeline.run()`: 以下の処理を追加
  - `web_entities.json`の保存処理
  - Web取得entityの`TextAttributedKnoweldgeGraph`への追加
  - 受け入れられたトリプルの`TextAttributedKnoweldgeGraph`への追加
  - 各iterationの最後に`TextAttributedKnoweldgeGraph.save_to_files()`呼び出し

### 互換性

- **後方互換性**: 既存の実験には影響なし（新規ファイルの追加のみ）
- **ファイル形式**: JSON形式で可読性が高い
- **サイズ**: entity数に比例（通常は数十KB程度）

### 重要な動作変更

**次のiterationでの学習にWeb取得情報が反映される**:

- 各iterationの最後に`entity2text.txt`、`entity2textlong.txt`、`train.txt`が更新される
- 次のiterationで埋め込みモデルを学習する際、これらのファイルが読み込まれる
- **Web取得されたentityのdescription情報が埋め込みモデル学習に使用される**（TAKG対応モデルの場合）
- Entity linkingによって既存entityに紐付けられた場合でも、元のweb:hash IDのentity情報は保持される

---

## 今後の拡張

### 分析ツールの追加

`web_entities.json`を活用した分析スクリプトの作成：

- Entity linkingの成功率計算
- Description品質の評価（文字数、情報量等）
- Source URLの分布分析（Wikipedia率、信頼性評価）

### Entity2text.txt形式での保存

TAKG標準形式（`entity2text.txt`, `entity2textlong.txt`）への変換機能：

```python
def save_web_entities_as_takg(web_entities_dict, output_dir):
    """web_entities.jsonをTAKG標準形式で保存"""
    with open(output_dir / "web_entity2text.txt", "w") as f:
        for eid, einfo in web_entities_dict.items():
            f.write(f"{eid}\t{einfo['description_short']}\n")
    
    with open(output_dir / "web_entity2textlong.txt", "w") as f:
        for eid, einfo in web_entities_dict.items():
            if einfo.get('description'):
                f.write(f"{eid}\t{einfo['description']}\n")
```

### Entity Linking結果の可視化

`web_entities.json`と`web_provenance.json`を組み合わせた可視化：

- Web entity → KG entityのリンク図
- 同一KG entityに複数のweb entityがリンクされているケースの検出

---

## 参考情報

### 関連コード

- `simple_active_refine/knoweldge_retriever.py`:
  - `Entity`: entity情報のデータモデル
  - `LLMKnowledgeRetriever.retrieve_knowledge_for_entity()`: Web取得の実装
  - `KnowledgeRefiner.find_same_entity()`: Entity linking実装

### 設定パラメータ

- `ArmPipelineConfig.candidate_source`: "web"に設定でWeb取得モード
- `ArmPipelineConfig.web_enable_entity_linking`: Trueでentity linking有効化

---

## 結論（観測→含意→次アクション）

### 観測

- Web取得で得られた entity のテキスト属性（label/description等）が失われる問題に対し、`web_entities.json` として iteration 単位で永続化する実装を追加した。
- 取得した entity と受け入れトリプルを `TextAttributedKnoweldgeGraph` に統合し、`entity2text*.txt` / `train.txt` へ反映して次iterationへ継承する導線を作った。
- entity linking が成功した場合は `linked_to` として紐付け先KG IDを記録できる。

### 含意

- Web取得は「トリプル」だけでなく「テキスト属性」を含めて初めてTAKG（KG-FIT等）へ寄与し得るため、永続化は必須コンポーネントになる。
- provenance と entity 属性が揃うことで、失敗分析（linking不成立/description品質）や監査（どのURL由来か）が現実的になる。

### 次アクション

- `web_entities.json` を用いた分析（linking成功率、description品質、URL分布）を定期レポート化する。
- entity linking のマップ永続化（キャッシュ）や、同一web entityの再利用戦略を検討する。

