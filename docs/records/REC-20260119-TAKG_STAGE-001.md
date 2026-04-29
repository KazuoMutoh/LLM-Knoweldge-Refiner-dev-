# REC-20260119-TAKG_STAGE-001: TAKG移行（STAGE参照）に向けたアルゴリズム整理と統合方針

作成日: 2026-01-19  
最終更新日: 2026-01-19

更新日: 2026-01-19（既存KGEとの併用方針、STAGE公式実装のキャッシュ運用知見を追記）

注記（2026-01-19）: TAKGの埋め込み方式は **KG-FIT** を採用する方針へ変更したため、本ドキュメントは「STAGE参照の検討記録」として位置づける。最新の実装仕様は [docs/records/REC-20260119-TAKG_KGFIT-001.md](REC-20260119-TAKG_KGFIT-001.md) を参照。

## 1. 背景 / 問題意識

現状のパイプラインは、静的なKGE（PyKEENのTransE等）を中心に
- ルール抽出（AMIE+）
- ルール選択（Bandit/LLM-policy）
- ルールに基づくトリプル追加
- （必要に応じて）再学習・評価

を回している。

一方で、/app/docs/external/TAKGへの移行検討.md で述べられている通り、**構造（トポロジー）だけの埋込**では、追加トリプルにより局所的に過制約になりやすく、スコアやランキングが悪化するケースが観測される。

これを緩和するため、**Text-attributed Knowledge Graph (TAKG)** の要素（エンティティ/リレーションにテキスト属性を持たせ、埋込の自由度を増やす）を、まずは知識グラフ埋込の段から導入する。

本検討では、/app/docs/external/STAGE.md に記載の **STAGE（Simplified Text-Attributed Graph Embeddings）** を採用し、テキストエンコーダをファインチューニングせずに、既存のKGEループへ統合する方針を整理する。

## 2. ゴール / 非ゴール

### ゴール

- **TAKG（STAGE方式）を“追加導入”**し、既存の構造のみKGE（PyKEEN/TransE等）も継続利用できるようにする（併用/切替/比較を可能にする）。
- STAGEで使うテキスト埋め込みは **OpenAI `text-embedding-3-small`** を使用する。
- テキスト埋め込みは毎回API計算しない（**事前計算・キャッシュ**を前提にする）。
- 既存パイプライン（arm-run / retrain-eval / relation priors 等）への統合ポイントと必要なI/F変更を明文化する。
- baseline KGE と STAGE-KGE の評価（Hits@k/MRR/target score等）が同一手順で比較できるよう、成果物・I/Fを揃える。

### 非ゴール（本ドキュメントではやらない）

- 実装そのもの（コード変更、実験実行、性能評価）は行わない。
- 既存KGE（PyKEENベース）を撤去/置換しない（後方互換を維持する）。
- retrieval側（Chroma/BM25）をSTAGEに合わせて最適化する、などの追加改善はスコープ外。

## 3. 現状の依存点（埋込周りの「差し替え影響範囲」）

埋込の中心I/Fは `simple_active_refine/embedding.py` の `KnowledgeGraphEmbedding`。
主に以下の箇所から参照されている（代表例）：

- 学習/評価/再学習
  - `scripts/train_initial_kge.py`
  - `retrain_and_evaluate_after_arm_run.py`
  - `simple_active_refine/pipeline_concrete.py`
- 解析・事前計算
  - `simple_active_refine/relation_priors_compute.py`（エンティティ埋込行列を抽出して prior を作る）
- ルール抽出
  - `simple_active_refine/rule_extractor_impl.py`

このため、STAGE導入は「埋込の学習ロジック」だけでなく、**(a) モデル保存形式**、**(b) score_triples / evaluate の互換性**、**(c) entity embedding の取得**にも影響し得る。

## 4. STAGEの要点（本プロジェクト用の最小定義）

注意: STAGE（Zolnai-Lucas et al., 2024; arXiv:2407.12860）は **Text-Attributed Graph (TAG) のノード分類**を対象とした手法であり、
本プロジェクトが扱う **KGE（リンク予測）そのものを提案する論文ではない**。

一方で、STAGEが強調する「(1) 事前学習済みLLMを **凍結した埋め込み生成器**として使う」「(2) 埋め込み生成を **事前計算して再利用**する」「(3) 余計な多段の生成/学習を避け、下流の学習を単純化する」という設計思想は、本プロジェクトのTAKG移行（=テキスト属性を埋込に活かす）にとって重要な参照点になる。

この章では、まず **論文でのSTAGE（原典）** を整理し、その後に **本プロジェクトでのKGEへの転用（STAGE-inspired）** を明示的に別物として定義する。

### 4.0 STAGE論文（原典）のアルゴリズム要約

STAGE論文の中心は「TAGのノード（エンティティ）に紐づくテキスト属性を、凍結したLLMベースの埋め込みモデルでベクトル化し、それをノード特徴量として下流GNN（複数モデルのアンサンブル）を学習する」という **cascading** パイプライン。

論文中の処理（要約）は概ね次の通り。

1) **Text Embedding Retrieval（埋め込み生成）**
  - 各ノードのテキスト属性（例: title+abstract）を埋め込みモデルに入力し、ノード特徴量 $\mathcal{X}$ を得る。
  - 埋め込みモデルは基本的に **追加のファインチューニング無し（zero-shot / off-the-shelf）**。
  - 任意で「タスク説明（instruction）」を入力のprefixとして付与し、埋め込みをbiasする設定も評価されている（効果は限定的、と報告）。

2) **GNN Training（下流学習）**
  - 生成した $\mathcal{X}$ と隣接行列 $\mathcal{A}$ を入力として、複数のGNN/MLPモデルを学習する。
  - 予測はアンサンブル（平均）して最終予測とする。
  - スケーラビリティの観点から、拡散型（diffusion-pattern）GNN（例: SIGN）も検討されている。

3) **（別設定）Parameter-efficient Finetuning**
  - 本体（STAGEの主張）は「LLMを凍結して単純化」だが、追加実験としてPEFT（LoRA）による埋め込みモデル側の微調整も比較されている。

本プロジェクトにとって重要なのは、STAGEが「LLMを凍結した埋め込み生成器として使い、埋め込み生成をパイプラインの外（事前計算）に切り出し、下流学習のI/Oを単純化する」という設計を明確に推している点。

### 4.1 事前計算されたテキスト埋め込み

- 各エンティティ $e$ のテキスト属性（`entity2text.txt` / `entity2textlong.txt` 等）から、OpenAI `text-embedding-3-small` でベクトル $t_e \in \mathbb{R}^{d_t}$ を計算して保存する。
- **埋め込みは実験/反復で再利用**し、学習時にAPIは叩かない。

備考: ユーザ文面にある `text-embeddin-3-small` はタイポと思われ、正は `text-embedding-3-small`。

### 4.2 本プロジェクトでのSTAGE-inspired適用（KGEへの転用）

ここから先は **STAGE論文の対象タスク（TAGノード分類）を、KGE（リンク予測）へ転用するための本プロジェクト独自の設計**。
論文のSTAGEは「GNNへ入れるノード特徴量を良くする」アプローチだが、本プロジェクトでは「KGEモデルのentity representationを良くする」形に置き換えて統合する。

実装に落としやすい最小構成（案）は次の通り。

1) **テキスト→固定ベクトル化（事前計算）**
  - 各エンティティ $e$ のテキスト属性 $text(e)$ を作る（例: `entity2text.txt` + `entity2textlong.txt` の連結）。
  - OpenAI `text-embedding-3-small` を用いて $t_e \in \mathbb{R}^{d_t}$ を得る。
  - 学習中はAPIを呼ばず、$t_e$ は固定（freeze）とする。

2) **固定テキスト埋め込み→KGE空間への射影（学習）**
  - 射影層 $f_\theta: \mathbb{R}^{d_t} \to \mathbb{R}^{d}$ を学習する。
  - エンティティ埋め込みは

$$
\mathbf{v}_e = f_\theta(\mathbf{t}_e) \quad (\text{optionally } +\, \Delta_e)
$$

3) **関係埋め込みとスコア関数（学習）**
  - 関係埋め込み $\mathbf{r} \in \mathbb{R}^{d}$ を学習する。
  - 既存互換性が高いTransEスコア（または他Interaction）を採用する。

$$
score(h,r,t) = -\lVert \mathbf{v}_h + \mathbf{r} - \mathbf{v}_t \rVert
$$

4) **学習ループ（PyKEENのKGEと同様）**
  - 正例トリプルに対して負例（head/tail corruption）を生成。
  - 損失（margin ranking / logistic / BCE 等）は、既存KGEと同じ枠組みで選べる。
  - 勾配が流れるのは $f_\theta$ と $\mathbf{r}$（および採用した場合は $\Delta_e$）のみ。

この転用は「STAGEの“固定テキスト埋め込みを下流モデルに入れる”」という思想をKGEへ持ち込むものであり、論文から直接導かれる唯一の形ではない点に注意する。

### 4.3 STAGE公式実装から取り込む「運用上の要点」

STAGE公式リポジトリ（https://github.com/aaronzo/STAGE）は、本プロジェクトのKGE用途とタスクは異なるものの、**大規模ノード向けの埋め込み事前計算・保存・再利用**の実装が整理されている。ここから以下の運用パターンを取り込む。

- **memmap + float16 保存**: 巨大な埋め込み行列でも「逐次書き込み・部分読み出し」を可能にし、容量も削減する。
- **命名規約で“埋め込み条件”を一意化**: 埋め込み元モデル名、次元、instruction（タスク記述）有無などをファイル名/メタに含め、取り違えを防ぐ。
- **instruction（タスク記述）付き埋め込みの分岐**: ベースのテキストだけでなく「用途に応じた instruction を前置」する派生を別成果物として管理する（本プロジェクトでは任意）。

本プロジェクトでは、まずは **instruction無し（entity2text由来）** をデフォルトとし、必要性が見えた時点で instruction variant を追加可能な成果物仕様にする。

## 5.3 PyKEEN（v1.10.2）の実装要点（本プロジェクトが依存している点）

本リポジトリは `pykeen==1.10.2` を利用しており、現行の `KnowledgeGraphEmbedding` は以下のPyKEENの性質に依存している。

### A) 学習・評価の入り口は `pykeen.pipeline.pipeline`

- `pipeline(model=..., training=TriplesFactory, validation=..., testing=..., training_kwargs=..., optimizer=..., loss=..., ...)` により学習が走る。
- `model` には文字列（例: `"TransE"`）だけでなく、**ModelクラスまたはModelインスタンス**も渡せる（拡張モデルを差し込める）。

### B) モデルは `model.score_hrt(mapped_triples)` を持つ

- 本リポジトリのスコアリングは `KnowledgeGraphEmbedding.score_triples()` で `self.model.score_hrt(triples)` を直接呼んでいる。
- したがって、STAGE統合でも **PyKEEN `Model` として `score_hrt` が機能する**ことが必須。

### C) 表現（Representation）と相互作用（Interaction）の分離

PyKEENの設計として、概ね以下で構成される。

- **Representation**: entity/relation のベクトル化（埋め込み行列・MLP等）
- **Interaction**: $(h,r,t)$ のスコア関数（TransE/DistMult/ComplExなど）
- **Model**: これらを束ね、学習ループから呼ばれるAPI（score_hrt等）を提供

STAGEはまさに「entity representation を“固定テキスト埋め込み＋射影”に置き換える」問題なので、PyKEEN拡張の粒度としては **Representation層の差し替えが本質**になる。

### D) 本リポジトリの周辺コードが期待するもの

- [simple_active_refine/relation_priors_compute.py](simple_active_refine/relation_priors_compute.py) では、`model.entity_representations[0]` から全エンティティ埋め込み行列を取り出して X7 を計算する。
  - そのためSTAGE統合でも、`entity_representations[0]` が「全entityの埋め込みを返せる」ことが望ましい。
- `PipelineResult.save_to_directory()` が生成する `trained_model.pkl` と `training_triples/` を前提にロードしている。

以上より、STAGE統合は「PyKEENの training/evaluation/save の枠組みに乗りつつ、entity representation だけを差し替える」形が、最小変更で後方互換を満たす。

## 5. 事前計算（テキスト埋め込み）の成果物設計

### 5.1 入力

- `dir_triples/entity2text.txt`（必須に近い）
- `dir_triples/entity2textlong.txt`（任意、あれば結合して利用）
- （任意）`dir_triples/relation2text.txt`：現状はarm選択（LLM-policy）で既に利用。STAGE側で使うかは将来拡張。

### 5.2 出力（提案）

「学習時に高速ロードでき、かつ大規模でも破綻しない」形式に寄せる。

#### 推奨（大規模対応）: memmap（STAGE公式実装の方式を踏襲）

- `dir_triples/.cache/text_embeddings/entity_text_embeddings__text-embedding-3-small__dim{d_t}__v1.emb`
  - 形式: `np.memmap` 相当（dtype: float16 推奨、shape: `(num_entities, d_t)`）
  - 目的: 逐次生成・部分ロード・容量削減
- `dir_triples/.cache/text_embeddings/entity_text_embeddings__text-embedding-3-small__dim{d_t}__v1.meta.json`
  - 例: `{ "provider": "openai", "model": "text-embedding-3-small", "dim": d_t, "dtype": "float16", "format": "memmap", "entity_to_row": {"/m/abc": 0}, "text_sources": ["entity2text.txt", "entity2textlong.txt"], "preprocess": {"concat": "short+long", "empty_text": "label_only"}, "created_at": "..." }`

#### 代替（小規模/取り回し優先）: .npy

- `dir_triples/.cache/text_embeddings/entity_text_embeddings__text-embedding-3-small__dim{d_t}__v1.npy`
  - shape: `(num_entities, d_t)`
- メタJSONは上記と同様

補足:
- embedding APIの再現性（同一入力での完全一致）を厳密保証するのは難しいため、**キャッシュを正**とする。
- entity text の前処理（short/long結合、空文の扱い）をmetaに記録する。
- 追加の正規化（例: L2正規化）を行う場合は、その有無/方式も必ずmetaへ記録して固定する。

## 6. パイプライン統合方針（どの部分をどう変えるか）

大枠としては「既存 `KnowledgeGraphEmbedding` の利用箇所に対し、baseline（既存PyKEEN KGE）と STAGE-KGE の**両方を差し込める**ようにする」。

### 6.1 統合の選択肢

#### 案A: PyKEEN拡張としてSTAGEモデルを実装

- PyKEENのModel/Representation拡張で、
  - entity embedding を `f_\theta(t_e)` で生成
  - relation embedding は通常のEmbedding
  - 学習/評価/保存は PyKEEN pipeline を流用

長所:
- 既存の `KnowledgeGraphEmbedding.train_model()` / `evaluate()` / `score_triples()` の構造を最小変更で維持しやすい。
- `PipelineResult.save_to_directory()` の形式をそのまま使える。

短所:
- PyKEEN内部の拡張点理解が必要。初回実装コストがやや高い。

#### 案B: PyTorchでSTAGE+TransEを独自実装し、I/Fだけ互換にする

- `KnowledgeGraphEmbedding` と同等のメソッドを持つ `StageKnowledgeGraphEmbedding` を新設。
- 学習/負例サンプリング/評価（ranking）を独自実装。

長所:
- アルゴリズムの自由度が高い。

短所:
- PyKEENの評価器/データ処理を再実装する必要があり、保守コストが上がる。
- 保存形式互換を別途設計する必要。

本プロジェクトの現状（PyKEEN前提の周辺スクリプトが多い）から、初期は **案A（PyKEEN拡張）を第一候補**とする。

併用要件（既存KGEも使える）に対しては、以下を前提にする。

- baseline KGE（現行）: 何も変えずに動くこと
- STAGE-KGE（新規）: `embedding_backend=stage` を指定した時だけ有効化されること
- 評価/スコアリング: 同一の呼び出しI/Fで両backendを比較できること

#### 案Aの詳細: 「PyKEEN拡張としてSTAGEモデルを実装」する具体像

STAGE（固定テキスト埋め込み＋射影）をPyKEENに統合する場合、差し替えポイントは次の2つに集約できる。

1) **Entity Representation を“固定テキスト埋め込み＋射影”にする**
2) **Interaction は既存の TransE を再利用する（=スコア関数互換）**

これにより、学習・評価・保存・ロード・`score_hrt`・RankBasedEvaluator がそのまま機能する。

以下、実装単位での設計を示す（まだ実装しないが、実装計画として具体化）。

補足: PyKEEN公式ドキュメントでは、モデル拡張は

- `ERModel` を **subclass** する
- `ERModel` を **instantiate** する（新規のModelクラスを作らない）

の両方が説明されている。本プロジェクトの「最小変更・後方互換」方針からは **instantiate方式（`model=ERModel` + `model_kwargs`）を第一候補**とする。

##### 1) 追加する（予定の）PyKEEN拡張コンポーネント

PyKEENには「固定特徴量を入力し、学習可能な変換を追加する」ための `TransformedRepresentation` がある。これはSTAGEの

- 固定テキスト埋め込み $t_e$（学習しない）
- 射影 $f_\theta$（学習する）

に対応付けられる。

したがって、**新規に必要な最小拡張は「事前計算済みOpenAI埋め込みを返すRepresentation」だけ**で、射影は `TransformedRepresentation` に委譲するのが最小実装になる。

- `PrecomputedTextRepresentation`（名称は仮）
  - 役割: entity id $e$ に対して、事前計算済みテキスト埋め込み $t_e \in \mathbb{R}^{d_t}$ を返す（学習しない）
  - 内部状態:
    - `text_embeddings`: shape `(num_entities, d_t)` の固定テンソル（float16推奨）
    - `max_id = num_entities`
    - `shape = (d_t,)`
  - forward:
    - `return text_embeddings[e]`

- `StageEntityRepresentation`（構成として）
  - 役割: $f_\theta(t_e)$ を返す（STAGEの本体）
  - 実体: `TransformedRepresentation(base=PrecomputedTextRepresentation, ...)`
  - 備考: 変換は線形や小さなMLPなど。ここが学習対象。

モデルは原則として `ERModel` を直接使う。

- STAGE-TransE（概念）
  - model: `ERModel`
  - interaction: `TransEInteraction`（既存）
  - entity representations: `StageEntityRepresentation`
  - relation representations: 通常の `Embedding(num_relations, d)`

これにより、`score_hrt` を含むPyKEEN標準APIが成立し、現行の `KnowledgeGraphEmbedding.score_triples()` の前提も満たせる。

##### 2) 事前計算キャッシュ（memmap/.npy）をモデルへ渡す方法

PyKEENの `pipeline()` は `model_kwargs` を通してモデルへ追加パラメータを渡せる。したがって、STAGEでは原則として **キャッシュのパス（＋meta）を渡し、モデル側でロード**する。

- `model_kwargs` の例（概念）:
  - `text_embedding_cache_path`: `.../entity_text_embeddings__... .emb or .npy`
  - `text_embedding_meta_path`: `... .meta.json`
  - `embedding_dim`: `d`（KGE側の次元）
  - `projection_type`: `linear|mlp`
  - （任意）`use_entity_delta`: `bool`

この方式にすると、学習スクリプト側（例: `scripts/train_initial_kge.py`）は“backend切替”のif分岐だけで済み、PyKEENに渡す引数の形は維持できる。

##### 3) 保存・ロード互換（本リポジトリの要件）

本リポジトリの `KnowledgeGraphEmbedding` は、`PipelineResult.save_to_directory()` が作る

- `trained_model.pkl`
- `training_triples/`（TriplesFactoryのバイナリ）

の存在を前提にロードする。よって、STAGE統合も同じ形式で保存できる必要がある。

特に重要なのは「巨大なテキスト埋め込みを `trained_model.pkl` に含めるか？」で、ここは段階導入する。

- **v1（実装容易・後方互換優先）**:
  - 学習時に text embedding をテンソルとしてモデルに保持し、そのままpickleに含める。
  - 長所: ロードが簡単（モデル単体で完結）。
  - 短所: `trained_model.pkl` が巨大化し得る。

- **v2（大規模向け・STAGE公式の思想に寄せる）**:
  - `trained_model.pkl` には「射影層・relation embedding（＋delta）」のみを保存。
  - text embedding は `model_dir/text_embeddings/` に別ファイルとして保存し、ロード時にパスから再オープンする。
  - 長所: モデルファイルの肥大化を抑えられる。
  - 短所: ロード時に追加ファイルが必要（運用規約が必須）。

初期段階は v1 で統合を成立させ、その後にv2へ移行する（必要性が見えたら）。

##### 4) `relation_priors_compute`（X7）互換

既存は `model.entity_representations[0]` から重み行列を取ろうとする。

- STAGEの場合は entity embedding が動的（射影）なので、
  - `entity_representations[0](indices=torch.arange(num_entities))` が `(n_entities, d)` を返す
  - という形に寄せるのが素直。
- 既存コードはフォールバックとして representation を呼び出しているため、STAGE側がこの呼び出しに対応していれば、最小変更で動く。

##### 5) `score_triples()` 互換

`KnowledgeGraphEmbedding.score_triples()` は `model.score_hrt(mapped_triples)` を呼ぶだけなので、STAGEモデルが PyKEEN Model として動けばそのまま互換。

unknown entity の扱いは、現行同様に `TriplesFactory` の mapping に存在しないものはスキップ（=現行挙動維持）とする。

### 6.2 既存コードへの具体的な変更点（設計レベル）

1) 埋込設定（config）
- `config_embeddings*.json` に `embedding_backend: "pykeen"|"stage"` 等を追加する。
- `stage` の場合は
  - `text_embedding_cache_path`（上記 .npy / meta）
  - `text_embedding_model_name`（検証用に `text-embedding-3-small` を記録）
  - `text_embedding_cache_format: "memmap"|"npy"`（ロード方式の明示）
  - `projection: {type: linear|mlp, out_dim: d}`
  - （任意）`use_entity_delta: bool` など

2) 学習
- `scripts/train_initial_kge.py` / `retrain_and_evaluate_after_arm_run.py` から呼ぶ学習関数を、backendに応じて分岐。
- ただし呼び出し側の構造は同じにし、成果物（model_dir）構造も可能な限り同じに寄せる。

（詳細）
- baseline: これまで通り `pipeline(model="TransE", ...)`
- stage: `pipeline(model=ERModel, model_kwargs={...interaction=TransE..., ...entity_representations=StageEntityRepresentation..., ...}, ...)` のイメージ

※ 実装段階では、`KnowledgeGraphEmbedding.train_model()` に `embedding_backend` を渡せるようにするか、`StageKnowledgeGraphEmbedding.train_model()` を追加して呼び出し側で分岐する。

3) 推論（triple scoring）
- `score_triples()` は、入力の(文字列)トリプルをID化してスコアリングする点は同じ。
- STAGEでは entity embedding が「固定テキスト埋込＋変換」で得られるため、unknown entity が混じった場合の扱いを明確化する（例：スキップ/例外/ダミー）。現状はスキップしてwarning。

4) エンティティ埋込行列の取得（relation priors など）
- `simple_active_refine/relation_priors_compute.py` は `kge.model.entity_representations[0]...` のような形に依存する可能性がある。
- STAGE化した場合でも、
  - 「全エンティティの埋込ベクトルを (n_entities, d) で返す」
  - 「entity_to_id と同じ順序」
  を保証するヘルパI/Fを用意する（例：`kge.get_entity_embedding_matrix()`）。

5) TextAttributedKnoweldgeGraph（既存）の位置付け
- `simple_active_refine/knoweldge_retriever.py` の `TextAttributedKnoweldgeGraph` は、ChromaDB/BM25用にOpenAI埋込をその場で計算している。
- STAGE導入後は「**学習用埋込キャッシュ**」と「**検索用ベクトルDB**」の責務を分離し、
  - 学習は .npy キャッシュを優先
  - 検索は必要ならChromaを使う
  とする（混線すると、学習のたびにAPI実行してしまう）。

6) 併用（baseline KGEとSTAGE-KGE）の運用
- 反復精錬（arm-run）や再学習（after）を回す際に、backendを切り替えて比較できるようにする。
- 具体例:
  - baseline KGE で arm-run を回し、after評価だけ STAGE-KGE に切り替えて「スコア悪化が緩和されるか」を見る
  - 逆に STAGE-KGE で学習し、baseline KGE で再評価して「既存指標での一貫性」を確認する

※ どちらが本番backendになるかは、この比較から決める（本ドキュメントでは決めない）。

## 7. 実装タスク分解（次フェーズで実装するときのTODO）

- STAGE用のテキスト埋込事前計算スクリプトを追加
  - 入力: entity2text(+long)
  - 出力: memmap `.emb` + meta.json（推奨）または `.npy` + meta.json
  - OpenAI `text-embedding-3-small` 固定
- PyKEEN拡張のSTAGEモデル（entity representationが固定テキスト埋込由来）を実装
- `KnowledgeGraphEmbedding` を「backend切替可能」または新クラス追加で、baseline KGE と共存させる（既存利用箇所は壊さない）
- 既存の呼び出しコード（train/retrain/evaluate/relation_priors_compute）を最小限修正
- 互換テスト（最低限）
  - 既存datasetで `train_model -> save -> load -> score_triples` が動く
  - relation_priors_compute が落ちない
  - 同一データで baseline と stage の before/after 指標が同一フォーマットで出力される

## 8. リスク / 懸念と対策案

- OpenAI埋込コスト
  - 対策: 必ず事前計算キャッシュ。差分計算（追加entityのみ）も設計。
- エンティティのテキスト欠損
  - 対策: 空文は「ラベルのみ」または特殊トークンで埋める。metaに方針を固定。
- PyKEEN拡張の複雑性
  - 対策: 初期はTransE相当の最小モデルに限定し、評価と保存形式互換を優先。

## 9. 参照

- /app/docs/external/TAKGへの移行検討.md
- /app/docs/external/STAGE.md
- STAGE公式実装（埋め込み事前計算・保存運用の参考）: https://github.com/aaronzo/STAGE
- `simple_active_refine/embedding.py`（現行KGE wrapper）
- `simple_active_refine/knoweldge_retriever.py`（TextAttributedKnoweldgeGraph：検索/テキスト管理）
