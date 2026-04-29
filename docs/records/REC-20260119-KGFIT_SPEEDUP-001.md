# REC-20260119-KGFIT_SPEEDUP-001: KG-FIT正則化計算の高速化検討

作成日: 2026-01-19
最終更新日: 2026-02-01

## 目的
KG-FITの正則化（anchor/cohesion/separation）計算がTransEに比べて学習時間を増加させているため、現状実装のボトルネックを特定し、対応策と効果見積を整理する。

## 対象範囲
- 実装: [simple_active_refine/kgfit_regularizer.py](../../simple_active_refine/kgfit_regularizer.py), [simple_active_refine/kgfit_representation.py](../../simple_active_refine/kgfit_representation.py)
- 実験結果: [models/20260119/kgfit_fb15k237_transe_full_seed/results.json](../../models/20260119/kgfit_fb15k237_transe_full_seed/results.json)

## 現状実装の概要
- `KGFitEntityEmbedding.forward()` 内で `KGFitRegularizer.update_with_indices()` を呼び出し、各バッチの埋め込み `x` に対して正則化項を計算。
- `update_with_indices()` は以下を毎バッチで実施:
  - anchor embedding と cluster center を `indices` で gather
  - `cosine_distance()` 内で **毎回** `normalize()` を実行
  - separation は `neighbor_clusters` に基づき **x を expand** して距離計算（B×K×D）

## ボトルネック整理
1. **正規化の重複計算**
   - `anchor_embeddings` と `cluster_centers` は固定値だが、毎バッチで `normalize()` を実行。
2. **separation のメモリ/計算負荷**
   - `x.unsqueeze(1).expand_as(neighbor_centers)` により巨大テンソルを作る。
   - 計算量は $O(B \cdot K \cdot D)$、さらに中間テンソルが大きい。
3. **indexの重複処理**
   - `indices` が重複する場合に同一エンティティの正則化を重複計算。
4. **正則化の全バッチ実行**
   - 常に正則化を全バッチで計算しており、学習初期/後期で同一コスト。

## 高速化対応案
### 1) 事前正規化キャッシュ
- **内容**: `anchor_embeddings` と `cluster_centers` を初期化時にL2正規化し、bufferとして保持。
- **変更点**: `cosine_distance()` での `normalize()` を削減。`x` のみを1回正規化。
- **効果見積**: 正則化の距離計算コストを **~1.5-2.0x** 改善（正規化2回分削減）。

### 2) separationのベクトル化（matmul）
- **内容**: `x_norm` と `neighbor_centers_norm` の内積を `einsum` / `bmm` で計算し、expandを回避。
- **変更点**: `x_exp = x.unsqueeze(1).expand_as(...)` を廃止。
- **効果見積**: 中間テンソル削減で **メモリ削減 + 速度 ~1.2-1.5x**。

### 3) `unique=True` の明示（重複削減）
- **内容**: entity representation で `unique=True` を設定し、同一エンティティに対する正則化を1回にする。
- **効果見積**: バッチ内重複率に依存（例: 30%重複なら正則化コスト **~1.4x** 改善）。

### 4) separation対象の削減（近傍Kの縮小）
- **内容**: `neighbor_clusters` の上位Kを小さくする（例: K=10→5）。
- **効果見積**: separationコスト **線形削減**（K半減で ~2x）。
- **補足**: 精度影響の検証が必要。

### 5) 正則化の間引き（N stepごと）
- **内容**: `update_with_indices()` を `step % N == 0` のみ実行。
- **効果見積**: N=2で **~1.8x**、N=4で **~3x**。
- **補足**: 収束性・性能低下リスクがあるため、段階的適用。

### 6) mixed precision（float16/bfloat16）
- **内容**: 正則化用テンソルのみ低精度にし、距離計算を高速化。
- **効果見積**: GPUで **~1.2-1.5x**。
- **補足**: 数値安定性・再現性の確認が必要。

## 効果の総合見積
- 正則化計算の比率が全体の **30-50%** 程度の場合:
  - (1)+(2)+(3)で **~1.5-2.5x** の正則化高速化 → 全体で **~15-40%** 短縮
  - (4)追加で **~さらに10-20%** 短縮が見込める
- 学習1epochが ~140s → **~85-120s/epoch** が現実的レンジ（目安）

## プロファイリング結果の分析（2026-01-19 実測）
- 実測ログ: [models/20260119/kgfit_fb15k237_transe_full_seed_fast_profile/training.log](../../models/20260119/kgfit_fb15k237_transe_full_seed_fast_profile/training.log)
- `KGFitRegularizer profile` の平均は **約1.1ms/step**（steps=200〜17000の範囲で安定）。
- ログ上の training は `cuda:0` で進行しており、正則化は **GPU側**で完結していると解釈できる。
   - 実装も `anchor_embeddings` / `cluster_centers` を `register_buffer()` しており、**CPU⇔GPU転送を都度行うコードは存在しない**。
   - したがって「正則化のたびにCPU転送が発生している」可能性は低い。
- 近傍K（論文のm）: 補遺D.1にて **m=5** と明記（grandparentを祖先に設定し、最大5近傍クラスタを探索）。

### 追加で高速化が可能な箇所
1. **separation計算のK削減**
    - 近傍Kは計算量が $O(B\cdot K\cdot D)$ で支配的。Kを削減すれば確実に速度改善。
    - **論文のK設定は未調査**。外部資料の確認が必要。
2. **profile値の安定性**
    - 1.1ms/step前後で頭打ちのため、次のボトルネックは**正則化以外**（負例サンプリング/スコア計算）に移っている可能性が高い。

## Evaluationの高速化方針
- 現状、学習時の評価が重い（例: 719s @ FB15k-237）。
- **候補案**:
   1. **評価のslice/batch縮小**: `pipeline` に `evaluator_kwargs` を渡し、`batch_size`/`slice_size` を抑える。
   2. **評価頻度の削減**: `evaluation_kwargs` で epoch 毎評価を抑制（例: `evaluation_batch_size`/`evaluation_frequency`）※PyKEEN仕様要確認。
   3. **フィルタリング無効**: `filtered=False` で高速化（指標は変わる）。
   4. **評価対象の縮小**: test/validのサブセットでスモーク評価し、本評価は後段で実行。

## 追跡事項
- 近傍Kの**論文設定値**を確認し、Kを変えた速度/精度トレードオフを追記する。
- 高速化適用後の Hits/MRR 変動を、同一seedで再評価する。
- `results.json` の `times.training` を比較して改善率を算出する。

## 推奨アクション（段階適用）
1. **低リスク**: 事前正規化キャッシュ + matmul化 + unique=True
2. **中リスク**: 近傍K削減（精度比較が必要）
3. **高リスク**: 正則化間引き / mixed precision

---

## 結論（観測→含意→次アクション）

### 観測

- KG-FITの学習時間増の主要因として、正則化（anchor/cohesion/separation）計算における正規化の重複・separationのexpandによる巨大テンソル・index重複などが整理された。
- プロファイル上、正則化はGPU側で完結しており、CPU⇔GPU転送が支配的という兆候は低い。

### 含意

- 速度改善は「実装の無駄（重複正規化、expand）」を潰す低リスク施策と、「neighbor_k削減/間引き」など性能へ影響し得る施策を分離して進めるのが安全。
- 正則化を最適化して頭打ちになった場合、次のボトルネックは負例サンプリング/スコア計算/評価に移る可能性が高い。

### 次アクション

- 低リスク施策（事前正規化キャッシュ + matmul/einsum化 + `unique=True`）を先に適用し、`times.training` とプロファイルで改善率を定量化する。
- 近傍Kの値をスイープし、速度/精度のトレードオフを記録して運用既定値を決める。
- 評価（filtered ranking）が重い場合は、評価頻度/設定/サブセット化でスモークと本評価を分離する。

## 参照
- [simple_active_refine/kgfit_regularizer.py](../../simple_active_refine/kgfit_regularizer.py)
- [simple_active_refine/kgfit_representation.py](../../simple_active_refine/kgfit_representation.py)
- [models/20260119/kgfit_fb15k237_transe_full_seed/results.json](../../models/20260119/kgfit_fb15k237_transe_full_seed/results.json)
