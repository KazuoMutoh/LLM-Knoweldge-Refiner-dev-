# REC-20260123-ARM_WEB_RETRIEVAL-001: Web（LLMKnowledgeRetriever）によるトリプル追加を統合ランナーに組み込む実装計画

作成日: 2026-01-23  
最終更新日: 2026-02-01

参照:
- 統合ランナー仕様: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](../rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- arm-run 実装仕様: [docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)
- 反復精錬（arm）設計（web取得の将来像含む）: [docs/records/REC-20260111-ARM-REFINE-001.md](REC-20260111-ARM-REFINE-001.md)
- 実装（Web取得）: [simple_active_refine/knoweldge_retriever.py](../../simple_active_refine/knoweldge_retriever.py)

---

## 0. 目的

従来は `train_removed.txt`（ローカル候補集合）から証拠トリプル（rule body に一致する triples）を取得していましたが、
**Web（LLMKnowledgeRetriever）から取得できるように**し、統合ランナーから **local / web を切り替え可能**にする。

最終的な利用経路:
- 統合ランナー: [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py)
  - Step3（arm refinement）に Web 取得をオプションとして組み込む

---

## 1. 現状整理（どこで train_removed を使っているか）

### 1.1 統合ランナー（run_full_arm_pipeline.py）
- `--candidate_triples` が必須で、通常 `train_removed.txt` を渡す
- Step3 で `ArmDrivenKGRefinementPipeline.from_paths(..., candidate_triples_path=...)` を呼ぶ

### 1.2 arm-run（ArmDrivenKGRefinementPipeline）
- 初期化時に `candidate_triples_path` の TSV を読み込んで `self.candidate_triples` に保持
- 反復ごとに `TripleIndex(list(self.kg_set) + self.candidate_triples)` を作り、
  `LocalArmTripleAcquirer` が `find_body_triples_for_head(...)` で evidence を抽出する

重要:
- 現状の Step3 は「候補集合 = ローカル（train_removed等）」を前提にしている
- ただし、`TripleIndex` は単なる triple の集合なので、**Web 由来 triples を混ぜる**こと自体は容易

---

## 2. 追加要件（Web 化にあたり守りたいこと）

切替要件:
1. デフォルトは現状維持（local=train_removed のみ）。既存実験が同じ引数で再現できること。
2. Web 取得を ON にした場合、Web を主に用いて反復が動くこと。
3. Web 取得が失敗した場合は、反復を落とさず「そのarm/targetはスキップ」等の best-effort で継続できること。

運用要件:
4. provenance（URL/取得時刻/クエリ等）を反復出力に保存できること。
5. キャッシュを持ち、同一 seed での再実行時に API 呼び出しを抑制できること。
6. 新規 entity ID が既存エンティティと衝突しないこと。
7. 追加トリプルのフィルタ（target predicate の混入、明らかなノイズ）を入れられること。

---

## 3. 方式検討（どこに組み込むか）

### 案A: 「候補集合」を Web に置き換える（最小変更・推奨）
- arm-run は従来どおり `TripleIndex` から evidence を抽出
- `candidate_triples`（候補集合）を Web 取得で作った triples に置き換える
- `LocalArmTripleAcquirer` と evaluator を基本的に変更せずに、取得ソースのみ差し替える

メリット:
- arm-run のロジック変更が小さい（“候補集合を差し替えるだけ”）

課題:
- Web 取得で使う「target サンプリング」と、acquirer 内部の「target サンプリング」を一致させる必要がある

### 案B: Web 用の ArmTripleAcquirer を別実装する（設計は綺麗だが変更が大きい）
- `LocalArmTripleAcquirer` と別に `WebArmTripleAcquirer` を作って切り替える
- さらに `CompositeArmTripleAcquirer` で合成する

本件（統合ランナーに組み込む第一歩）は案Aを採用し、必要になったら案Bへ拡張する。

---

## 4. 提案仕様（切替可能な I/F と設定）

### 4.1 `ArmPipelineConfig` に取得ソース設定を追加
- `acquisition_source: str = "local"`（`local|web`）
- `web_retrieval_enabled: bool = False`（`acquisition_source` の別表現でもよい）
- `web_retrieval_mode: str = "rule_body"`（`rule_body|entity_relation`）
  - `rule_body`: `LLMKnowledgeRetriever.retrieve_knowledge(target_triples, rules)` を使い、rule body を満たす triples を取得（推奨）
  - `entity_relation`: `retrieve_knowledge_for_entity(entity, relations)` を使い、(entity, relation, ?) 形式で拡張（既存の WebSearchTripleAcquirer 相当）
- `web_llm_model: str = "gpt-4o"`
- `web_use_web_search: bool = True`
- `web_cache_dir: Optional[str] = None`（既定: `<arm_run_dir>/web_cache/`）
- `web_max_triples_per_iteration: int = 200`（コスト/ノイズ制御）
- `web_max_targets_per_arm: int = 5`（Web に投げる seed 数上限）

後方互換:
- 既定値は「Web 無効」で現状と同じ挙動になる

### 4.2 反復内の target サンプリングを“パイプライン側”に集約
現状は `LocalArmTripleAcquirer.acquire(...)` が `target_triples` からサンプリングしている。
案Aを成立させるため、以下のどちらかを行う:

- 変更案（推奨）: `LocalArmTripleAcquirer.acquire(..., targets_by_arm: Optional[Dict[str, List[Triple]]] = None)` を追加
  - `targets_by_arm` が渡されたらそれを使用し、無ければ従来どおり内部サンプリング
  - これにより Web 取得と local 取得が同じ seed を共有できる

---

## 5. Web 取得の具体フロー（案Aの手順）

### 5.1 反復 1 回の処理フロー（概略）
1. arms 選択（現状どおり）
2. `targets_by_arm` を決定（パイプライン側でサンプリング、seed固定）
3. Web 取得（設定時のみ）
   - 入力: `targets_by_arm` と、arm 内の `rule_keys` から引いた `AmieRule`（body pattern）
   - 出力: `web_candidate_triples`（TSV triple list） + `provenance_by_triple`
4. `TripleIndex` を構築
  - `acquisition_source=local`: `TripleIndex(current_kg_triples + local_candidate_triples)`
  - `acquisition_source=web`: `TripleIndex(current_kg_triples + web_candidate_triples)`
5. evidence 抽出（LocalArmTripleAcquirer）
   - `find_body_triples_for_head(...)` は候補集合に Web が混ざっていても動作する
6. evaluator（現状どおり）
7. KG 更新（現状どおり: accepted evidence + incident triples）
8. 出力に provenance を追記

### 5.2 Web 取得の seed 設計（推奨）
- `rule_body` モードでは、target triple と rule body patterns をそのまま LLM に与え、
  **body を満たす triple 群（evidence）**を返させる
- 取得対象は原則「body predicate」のみ（target predicate は混ざっても棄却/別枠）

---

## 6. ID設計（Web由来 entity の衝突回避）

`LLMKnowledgeRetriever` は現状 `e1, e2, ...` のようなローカルIDを振っているため、
そのままでは「反復を跨いだ衝突」や「既存KGとの衝突」が起き得る。

提案:
- Web由来 entity id の正準形を導入し、以下のいずれかで安定化する
  - `web:<sha1(label + source_url)>`（推奨: 再現性と重複排除が容易）
  - `web:iter<k>:<seq>`（簡単だが重複排除が難しい）

この変換は arm-run 側（Web取得アダプタ層）で行い、KGに入る triple は常に「衝突しないID」を使う。

---

## 7. 出力仕様（監査・再現のため）

各 iteration ディレクトリ（`arm_run/iter_k/`）に追加:
- `web_retrieved_triples.tsv`（反復で取得した Web triples の生リスト）
- `web_provenance.json`（`{ "(s,p,o)": {source,url,query,retrieved_at,...} }` のような辞書 or JSONL）
- `diagnostics.json` に Web 取得統計を追記
  - `web_retrieved_total`, `web_used_total`, `web_cache_hit_total`, `web_errors_total`

既存ファイル（互換維持）:
- `accepted_evidence_triples.tsv`, `accepted_added_triples.tsv` は従来どおり

---

## 8. 統合ランナー（run_full_arm_pipeline.py）への導線（CLI案）

既存の `--candidate_triples` は残し、デフォルト挙動を維持する。

追加引数案:
- `--candidate_source {local,web}`（既定 `local`）
  - `local`: 従来どおり `--candidate_triples` を使用
  - `web`: Step3（arm-run）では Web 取得で候補集合を構築（ただし Step2 の arm 生成で `--candidate_triples` が必要なため、統合ランナーでは当面必須のまま運用）
- `--web_llm_model`（既定 `gpt-4o`）
- `--disable_web_search`（web_search_preview を使わず通常LLMにフォールバック）
- `--web_cache_dir`（既定 `<run_dir>/arm_run/web_cache`）
- `--web_max_triples_per_iter` / `--web_max_targets_per_arm`

実装上の注意:
- OpenAI APIキーが無い場合は、`candidate_source=web` のとき明示的にエラー or 警告して local にフォールバック

---

## 9. 実装ステップ（DoD付き）

### Step 1: arm-run に Web 取得アダプタを追加
- `ArmDrivenKGRefinementPipeline.run()` 内で、反復ごとに Web 候補 triples を生成し `TripleIndex` に使用できる

DoD:
- `candidate_source=local` で既存実験が同一結果（または少なくとも同一I/O形）で走る
- `candidate_source=web` で `web_retrieved_triples.tsv` が出る

### Step 2: target sampling の外出し（共通 seed）
- `LocalArmTripleAcquirer` に `targets_by_arm` 受け取りを追加（後方互換）
- Web 取得と local 取得が同じ seed を使う

DoD:
- `targets_by_arm` を固定すると再現可能（反復の乱数固定）

### Step 3: entity id 正規化 + provenance 保存
- Web 由来 entity id を正準化（衝突回避）
- provenance を `iter_k/web_provenance.json` に保存

DoD:
- Web triples の subject/object が衝突しない
- provenance が accepted_evidence に紐付けられる（少なくとも取得時点で保存される）

### Step 4: 統合ランナーにCLIを追加
- `run_full_arm_pipeline.py` から Step3 の config を渡せる

DoD:
- `python run_full_arm_pipeline.py --help` に Web オプションが出る

---

## 10. テスト計画

- unit: Web取得アダプタを「ダミー retriever（固定レスポンス）」でテストし、ネットワーク不要で動く
- integration: 小さな toy dataset で `candidate_source=web` を 1 iteration だけ回し、
  `web_retrieved_triples.tsv` と既存の `accepted_*` が出ることを確認

---

## 11. リスクと対策

- コスト/レート制限: `web_max_*` とキャッシュを必須化
- ノイズ混入: target predicate の混入排除、URL必須、既存KGとの重複除去
- 再現性: entity id 正規化（hash）と、キャッシュ保存（生レスポンス）
- 失敗耐性: web失敗時は local のみで続行

---

## 結論（観測→含意→次アクション）

### 観測

- 既存の arm-run（候補集合→TripleIndex→evidence抽出）を崩さず、「候補集合をWeb由来 triples に差し替える」案Aが最小変更で統合可能と整理した。
- Web運用に必須な要件（provenance保存、キャッシュ、stable web ID、ノイズ混入対策、失敗時のbest-effort継続）を、統合ランナーのI/Fとiter出力に落とし込んだ。

### 含意

- Web取得は、単なる候補追加ではなく「再現性・監査性（何をどこから取ったか）」が成果物設計の中心になる。
- entity ID衝突回避（stable web ID）と、target samplingの共有は、witness成立と実験再現の両方に直結するため優先度が高い。

### 次アクション

- `candidate_source=web` の反復で `web_retrieved_triples.tsv` / `web_provenance.json` / `diagnostics` が確実に出る最小統合を先に完了させる。
- target sampling をパイプライン側に集約し、local/webが同一seedで比較できるようにする。
- stable web ID と provenance の仕様を固定し、後続の entity linking / 永続化（web_entities.json）へ接続する。
