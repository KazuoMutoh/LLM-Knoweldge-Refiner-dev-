# RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001: KG-FIT正則化（anchor/cohesion/separation）高速化と運用標準

作成日: 2026-01-19

## 目的
KG-FITバックエンドの学習時間に影響する「正則化（anchor/cohesion/separation）」について、
- 計算の定義
- 計算量の支配項
- 実装上の最適化ルール（キャッシュ/ベクトル化/重複排除）
- 近傍K（neighbor_k）の標準値とチューニング規約
- プロファイル方法と評価高速化の運用
を **再実装可能な粒度**で定義する。

参照（検討記録）:
- [docs/records/REC-20260119-KGFIT_SPEEDUP-001.md](../records/REC-20260119-KGFIT_SPEEDUP-001.md)

## 対象コード
- [simple_active_refine/kgfit_regularizer.py](../../simple_active_refine/kgfit_regularizer.py)
- [simple_active_refine/kgfit_representation.py](../../simple_active_refine/kgfit_representation.py)
- KG-FIT統合（neighbor_k設定の流れ）: [simple_active_refine/embedding.py](../../simple_active_refine/embedding.py)

## 正則化の定義（実装準拠）

前提:
- バッチの entity embedding: $x \in \mathbb{R}^{B\times D}$
- entity id（TriplesFactoryのentity_to_idに対応）: `indices`（長さB）
- anchor embeddings（テキスト埋め込み初期値）: $a \in \mathbb{R}^{|E|\times D}$
- entity-to-cluster index: $c_e \in \{0,\dots,|C|-1\}^{|E|}$
- cluster centers: $\mu \in \mathbb{R}^{|C|\times D}$
- neighbor clusters（各クラスタの近傍クラスタ index）: $N \in \mathbb{N}^{|C|\times K}$

使用距離:
- cosine distance $d(u,v)=1-\langle \hat{u}, \hat{v}\rangle$（$\hat{u}$ はL2正規化）

### 1) anchor loss
- `anchors = a[indices]`
- $\mathcal{L}_{anc} = \mathrm{mean}_b\, d(\hat{x}_b, \hat{anchors}_b)$

### 2) cohesion loss
- `cluster_ids = c_e[indices]`
- `centers = μ[cluster_ids]`
- $\mathcal{L}_{coh} = \mathrm{mean}_b\, d(\hat{x}_b, \hat{centers}_b)$

### 3) separation loss（margin）
- `neighbor_ids = N[cluster_ids]` （shape = B×K）
- `neighbor_centers = μ[neighbor_ids]` （shape = B×K×D）
- cosine similarity は内積で計算し、距離に変換
- $\mathcal{L}_{sep} = \mathrm{mean}_{b,k}\, \max(0, m - d(\hat{x}_b, \hat{neighbor\_centers}_{b,k}))$

総和:
- $\mathcal{L} = w_a\mathcal{L}_{anc} + w_c\mathcal{L}_{coh} + w_s\mathcal{L}_{sep}$

## 計算量とボトルネック
支配項:
- separation が $O(B\cdot K\cdot D)$
- かつ中間テンソル（B×K×D）を扱うため、GPUメモリ/帯域に効く

典型的な遅さの原因:
- 毎ステップの `normalize()` の重複（固定テンソルまで毎回正規化）
- separation で `expand` を使い巨大テンソルを生成
- `indices` の重複により同一entityの正則化をバッチ内で繰り返す

## 実装上の高速化ルール（必須）

### ルール1: 固定テンソルの事前正規化
- anchor embeddings と cluster centers は学習中に固定（バッファ）であり、初期化時にL2正規化して保持する
- 学習ステップでは `x` のみを正規化する

根拠（実装）:
- [simple_active_refine/kgfit_regularizer.py](../../simple_active_refine/kgfit_regularizer.py) で `register_buffer()` し、初期化時に `F.normalize()` 済み

### ルール2: separationはeinsum/bmmで計算する
禁止:
- `x.unsqueeze(1).expand_as(neighbor_centers)` のような明示的 expand

推奨:
- `cosine_sim = torch.einsum("bd,bkd->bk", x_norm, neighbor_centers_norm)`
- 距離 `dist = 1 - cosine_sim` を介して margin loss

### ルール3: `unique=True` で重複indexを潰す
- entity表現は `unique=True` を標準とし、Embedding内部で `indices.unique(return_inverse=True)` を使って重複を除去する
- その後、元の順序へ戻すため `inverse` で復元する

根拠（実装）:
- [simple_active_refine/kgfit_representation.py](../../simple_active_refine/kgfit_representation.py)

## 近傍K（neighbor_k）の標準とチューニング

### 標準値
- `neighbor_k=5` を標準とする
  - 近傍数は separation の計算量に線形で効く
  - KG-FIT論文の設定（m=5）とも整合

### チューニング規約
- `neighbor_k` を変更した場合、必ず以下を同一seedで比較する
  - 学習時間（epoch time / steps/sec）
  - Hits@k / MRR
  - target_triples の平均スコア変化（after-before）

推奨の探索順序:
- 5 → 3 → 1（段階的）

実装上の注意:
- `neighbor_clusters.json` にはクラスタごとの近傍候補（最大k）を持たせておき、学習側で `neighbor_k` に応じて trim する
- 学習側の実装は `max_k = min(neighbor_k, max_len_in_json)` としてテンソルを構築する

## プロファイリング（標準運用）

### 計測対象
- 正則化の forward/更新時間（GPU同期を含める）

### 実装準拠の計測方法
- `KGFitRegularizerConfig.profile=True` のとき、`profile_every` stepごとにログを出す
- GPUの場合は `torch.cuda.synchronize()` を挟んで wall-clock を測る

注意:
- `synchronize` は学習全体を遅くするため、通常運用では `profile=False` にする

## 評価（evaluation）高速化の運用
KG-FITに限らず、PyKEEN評価が学習時間を支配する場合がある。
この場合、まず以下を検討する。

- evaluator の `batch_size` / `slice_size` を小さくしてメモリ破綻を避けつつ throughput を上げる
- 評価頻度を落とす（epochごと評価を避ける）
- スモーク時は test/valid のサブセットで回す

備考:
- 評価の最適化は「学習損失そのもの」ではなく、実験運用時間を短縮する目的で行う

## 受け入れ基準
- `neighbor_k=5` で学習が安定して完走し、正則化ログが想定通り出る（profile時）
- `neighbor_k` を変更してもクラッシュしない（近傍配列長が足りない場合は安全にtrimされる）
- `unique=True` により `indices` が重複しても計算が一致する（同一埋め込みに対し同一loss）

## 変更履歴
- 2026-01-19: 新規作成（正則化高速化と運用標準を定義）
