# RULE-20260119-TAKG_KGFIT-001: TAKG（KG-FIT）KGEバックエンドの標準仕様

作成日: 2026-01-19

## 目的
本プロジェクトのKGE（Knowledge Graph Embedding）に、Text-attributed Knowledge Graph (TAKG) の情報を取り込むための **KG-FITバックエンド**を追加導入した。
このドキュメントでは、
- 事前計算成果物（テキスト埋め込み/seed階層）
- 学習I/F（`embedding_backend="kgfit"` と `kgfit_config`）
- 性能・再現性の運用ルール
を **設計標準（rules）**として定義する。

## スコープ
- 対象コード:
  - [simple_active_refine/embedding.py](../../simple_active_refine/embedding.py)
  - [simple_active_refine/kgfit.py](../../simple_active_refine/kgfit.py)
  - [simple_active_refine/kgfit_precompute.py](../../simple_active_refine/kgfit_precompute.py)
  - [simple_active_refine/kgfit_hierarchy.py](../../simple_active_refine/kgfit_hierarchy.py)
  - [simple_active_refine/kgfit_regularizer.py](../../simple_active_refine/kgfit_regularizer.py)
  - [simple_active_refine/kgfit_representation.py](../../simple_active_refine/kgfit_representation.py)
  - [scripts/compute_kgfit_text_embeddings.py](../../scripts/compute_kgfit_text_embeddings.py)
  - [scripts/build_kgfit_seed_hierarchy.py](../../scripts/build_kgfit_seed_hierarchy.py)
  - [scripts/train_initial_kge.py](../../scripts/train_initial_kge.py)
- 対象外:
  - LLM-guided hierarchy refinement (LHR) の導入（本リポジトリでは seed 階層までを標準とする）
  - `project` 方式（学習可能射影による圧縮）の実装（現状 loader は `full|slice` のみ）

## 背景（要点）
KG-FITは「LLMをKGEに合わせて微調整」ではなく、「テキスト埋め込み（固定）と階層（事前計算）を用いてKGE側を微調整」する枠組み。
本プロジェクトでは、TAKG化の第一歩として **entityテキスト埋め込み + seed階層制約**をKGEへ統合する。

参照（検討記録）:
- [docs/records/REC-20260119-TAKG_KGFIT-001.md](../records/REC-20260119-TAKG_KGFIT-001.md)

## 標準ディレクトリと成果物
KG-FITの事前計算成果物は、データセットディレクトリ（`dir_triples`）直下の `.cache/kgfit/` に置く。

### 必須成果物（学習に必要）
- `entity_name_embeddings.npy`
- `entity_desc_embeddings.npy`
- `entity_embedding_meta.json`
  - `entity_to_row` を含む（entity文字列 -> 行番号）

ファイル仕様（再実装のための厳密条件）:
- `entity_name_embeddings.npy` / `entity_desc_embeddings.npy` は numpy 2D array（shape = `(n_entities, d_text)`）
- `entity_embedding_meta.json` は JSON object で、少なくとも次を含む
  - `entity_to_row`: JSON object（`{ "<entity>": <row_index>, ... }`）
  - 推奨メタ: `provider`, `model`, `dim`, `dtype`, `text_sources`, `created_at`
- `entity_to_row` の行番号は 0-indexed
- `entity_to_row` は **全行を完全にカバー**していること（欠損行がある場合はエラー）

### 必須成果物（seed階層に必要）
- `hierarchy_seed.json`
- `cluster_embeddings.npy`
- `neighbor_clusters.json`

ファイル仕様（seed階層）:
- `hierarchy_seed.json` は JSON object で、少なくとも次を含む
  - `entity_ids`: entity文字列の配列（長さ = `n_entities`）
  - `labels`: 各entityのクラスタラベル配列（長さ = `n_entities`）
  - `cluster_labels`: 使用されたクラスタラベル一覧（実装はこれを `cluster_centers` の index に対応づける）
  - `tau_opt` と探索パラメータ（`tau_min/tau_max/tau_steps`）は再現性のため推奨
- `cluster_embeddings.npy` は numpy 2D array（shape = `(n_clusters, d_embed)`）
- `neighbor_clusters.json` は JSON object（`{ "<cluster_index>": [<neighbor_cluster_index>, ...], ... }`）
  - ここでの `<cluster_index>` は `cluster_embeddings.npy` の行 index（0..n_clusters-1）を想定

備考:
- `entities.txt` が `dir_triples` にある場合、埋め込みの entity 行順序は `entities.txt` を優先して確定する（無い場合は entity2text のキーをソートして確定）。

重要（entity順序のルール）:
- 事前計算時に採用した entity 順序と、学習時に使用する entity mapping（`TriplesFactory.entity_to_id`）が一致していないと、誤対応で学習が破綻する。
- 本実装は学習時に `entity_to_row` を使って entity->行を解決し、`TriplesFactory.entity_to_id` の id 順に並べ替えてから読み込む。
  - よって、事前計算成果物の entity 行順序は自由だが、**`entity_to_row` が正しく完全**である必要がある。

## 事前計算（標準手順）

### 1) テキスト埋め込みの事前計算
- 実行スクリプト: [scripts/compute_kgfit_text_embeddings.py](../../scripts/compute_kgfit_text_embeddings.py)
- 入力テキスト（デフォルト）:
  - `entity2text.txt`（name相当）
  - `entity2textlong.txt`（desc相当）

推奨:
- `--model text-embedding-3-small` を標準とする
- API呼び出しは事前計算時のみ（学習中に外部APIを叩かない）

標準コマンド（例）:
```bash
python3 scripts/compute_kgfit_text_embeddings.py \
  --dir_triples <DATASET_DIR> \
  --model text-embedding-3-small \
  --dtype float32 \
  --batch_size 128
```

### 2) seed階層の構築
- 実行スクリプト: [scripts/build_kgfit_seed_hierarchy.py](../../scripts/build_kgfit_seed_hierarchy.py)
- 仕様:
  - name/desc埋め込みを `reshape_strategy` に従い結合してクラスタリング
  - 階層クラスタは average linkage + cosine distance
  - `tau` はスキャンして silhouette score が最大となる値を採用
  - 近傍クラスタは中心ベクトルの cosine 距離で kNN

推奨（論文/性能の観点）:
- `--neighbor_k 5` をデフォルトとする（学習時のseparationコストが $O(B\cdot K\cdot D)$ のため、Kは小さく保つ）

標準コマンド（例）:
```bash
python3 scripts/build_kgfit_seed_hierarchy.py \
  --dir_triples <DATASET_DIR> \
  --reshape_strategy full \
  --neighbor_k 5
```

## 学習I/F（標準仕様）

### 埋め込み設定ファイルのキー
初期KGE学習のCLI（[scripts/train_initial_kge.py](../../scripts/train_initial_kge.py)）は、embedding config JSON を読み込み、以下のキーを解釈する。

- `model`: 例 `"TransE"`
- `embedding_backend`: `"pykeen" | "kgfit"`
- `kgfit`: KG-FITバックエンド専用設定（任意）

### `embedding_backend="kgfit"` の前提
- `dir_triples` が必須（built-in dataset名では動かない）
- 現状は `model="TransE"` または `model="PairRE"` をサポート（[simple_active_refine/embedding.py](../../simple_active_refine/embedding.py)）
- entity埋め込みは `PretrainedInitializer` により、事前計算済みテキスト埋め込みから初期化される

注意（model名の表記）:
- 実装は `model` を lower-case 化して扱うため、`"TransE"` / `"transe"` や `"PairRE"` / `"pairre"` は同義
- rules としては config の表記ゆれを減らすため、`"transe"` / `"pairre"`（lower-case）を推奨する

重要（embedding_dimの扱い）:
- KG-FITバックエンドでは、entity embedding の次元は事前計算済みテキスト埋め込み（`pretrained.shape[1]`）で決まる
- したがって、`model_kwargs.embedding_dim` を指定しても KG-FIT側では基本的に利用されない（`reshape_strategy="slice"` の場合のみ `kgfit.embedding_dim` を使う）

実装上の構成（再実装用の具体）:
- PyKEEN の `ERModel` を直接構築する
  - entity representation: `KGFitEntityEmbedding`（`unique=True`）
  - relation representation:
    - TransE: `pykeen.nn.Embedding`（1本）
    - PairRE: `pykeen.nn.Embedding` を2本（$r_h, r_t$）
  - interaction:
    - TransE: `TransEInteraction(p=scoring_fct_norm)`
    - PairRE: `PairREInteraction(p=scoring_fct_norm)`
- 追加損失は `KGFitEntityEmbedding.forward()` が `KGFitRegularizer.update_with_indices(x, indices)` を呼ぶことで `regularization_term` に加算され、PyKEEN側の `collect_regularization_term` 経由で学習損失に合流する

### PairRE（KG-FIT）設定例（標準）
PairREを使用する場合、TransEとの差分は **`model` と relation representation**のみであり、KG-FIT（テキスト埋め込み/階層/正則化）の設定は同様でよい。

embedding_config の例（PairRE, 最小）:
```json
{
  "model": "pairre",
  "embedding_backend": "kgfit",
  "kgfit": {
    "reshape_strategy": "full",
    "hierarchy": {"neighbor_k": 5},
    "regularizer": {
      "anchor_weight": 0.5,
      "cohesion_weight": 0.5,
      "separation_weight": 0.5,
      "separation_margin": 0.2
    }
  },
  "model_kwargs": {"scoring_fct_norm": 1}
}
```

設定ファイル（実物）:
- `config_embeddings_kgfit_pairre_fb15k237.json`: FB15k-237向けのPairRE設定サンプル（パス固定）

### `kgfit` 設定スキーマ（最小）
`kgfit` は次を想定する（不明キーは無視され得るが、標準としてこの形に揃える）。

- `reshape_strategy`: `"full" | "slice"`
- `embedding_dim`: `slice` のとき必須（偶数）
- `paths`（任意）:
  - `name_embeddings`: 既定 `.cache/kgfit/entity_name_embeddings.npy`
  - `desc_embeddings`: 既定 `.cache/kgfit/entity_desc_embeddings.npy`
  - `meta`: 既定 `.cache/kgfit/entity_embedding_meta.json`
- `hierarchy`（任意）:
  - `hierarchy_seed`: 既定 `.cache/kgfit/hierarchy_seed.json`
  - `cluster_embeddings`: 既定 `.cache/kgfit/cluster_embeddings.npy`
  - `neighbor_clusters`: 既定 `.cache/kgfit/neighbor_clusters.json`
  - `neighbor_k`: 学習時に使う近傍クラスタ数（既定 5）
- `regularizer`（任意）:
  - `anchor_weight`, `cohesion_weight`, `separation_weight`
  - `separation_margin`
  - `profile`, `profile_every`

embedding_config の例（最小）:
```json
{
  "model": "TransE",
  "embedding_backend": "kgfit",
  "kgfit": {
    "reshape_strategy": "full",
    "hierarchy": {"neighbor_k": 5},
    "regularizer": {
      "anchor_weight": 0.5,
      "cohesion_weight": 0.5,
      "separation_weight": 0.5,
      "separation_margin": 0.2
    }
  },
  "model_kwargs": {"scoring_fct_norm": 1}
}
```

設定ファイル（実物）:
- `config_embeddings_kgfit.json`: パスを省略し、`<dir_triples>/.cache/kgfit/` を前提にする汎用サンプル
- `config_embeddings_kgfit_fb15k237.json`: FB15k-237（/app/data/FB15k-237）向けにパスを固定したサンプル

実行（例）:
```bash
python3 scripts/train_initial_kge.py \
  --dir_triples <DATASET_DIR> \
  --output_dir <MODEL_OUT_DIR> \
  --embedding_config <EMBEDDING_CONFIG_JSON> \
  --num_epochs 100
```

### 正則化（実装上の標準）
KG-FITバックエンドの正則化は、`KGFitEntityEmbedding.forward()` から `KGFitRegularizer.update_with_indices()` を呼び、以下を計算する。
- anchor: テキスト埋め込み（事前計算）との cosine 近接
- cohesion: cluster center との cosine 近接
- separation: neighbor cluster centers から margin で分離

実装上の標準（性能）:
- anchor/center は `register_buffer()` で保持し、初期化時に L2 正規化してキャッシュする
- separation は `torch.einsum("bd,bkd->bk", ...)` を使い、`expand` による巨大テンソル生成を避ける
- entity表現は `unique=True` をデフォルトとし、バッチ内の重複 index による重複計算を削減する

重要（seed階層とクラスタ中心のindex整合）:
- `hierarchy_seed.json` の `labels` は **クラスタラベル（任意の整数）**であり、`cluster_embeddings.npy` の行 index とは一致しない可能性がある。
- 実装は `cluster_labels` を用いて `label -> cluster_index(0..n_clusters-1)` を構成し、
  `entity_cluster_indices[entity_id] = cluster_index` を作ることで整合を取る。
- `neighbor_clusters.json` は **cluster_index** を前提とする（`label` を使わない）。

参照（検討記録）:
- [docs/records/REC-20260119-KGFIT_SPEEDUP-001.md](../records/REC-20260119-KGFIT_SPEEDUP-001.md)

## 性能チューニング規約（運用ルール）

### 優先度: 低リスク → 中リスク → 高リスク
- 低リスク（まずやる）:
  - 事前正規化キャッシュ（実装済み）
  - separation の einsum/bmm 化（実装済み）
  - `unique=True`（実装済み）
- 中リスク:
  - `neighbor_k` を下げる（例: 5→3）。精度（Hits/MRR/target score）とのトレードオフを必ず比較する
- 高リスク:
  - 正則化の間引き（N stepごと）
  - mixed precision の強制（数値安定性・再現性に注意）

### 推奨デフォルト
- `neighbor_k=5`
- `regularizer.profile=False`（スモーク/プロファイル時のみ True）

## 受け入れ基準（再現性）
KG-FITバックエンドを使う実験は、以下が揃っていること。
- `.cache/kgfit/` 配下の成果物一式が run から参照可能
- embedding config に `embedding_backend="kgfit"` と `kgfit` 設定が明記されている
- `entity_embedding_meta.json` の `model/dim/dtype/text_sources/created_at` が記録されている

追加（PairREを使う場合）:
- PairREの学習スモークが通ること（ユニットテスト）: [tests/test_kgfit_pairre_backend.py](../../tests/test_kgfit_pairre_backend.py)

## 変更履歴
- 2026-01-19: 新規作成（KG-FITバックエンドの標準仕様を定義）
- 更新日: 2026-01-21（PairREサポートを追記）
- 更新日: 2026-01-23（PairRE運用の標準設定例と注意点、受け入れ基準を追記）
