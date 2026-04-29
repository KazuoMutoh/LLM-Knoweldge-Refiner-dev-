# REC-20260121-KGFIT_PAIRRE_IMPLEMENTATION_PLAN-001: KG-FITバックエンドへのPairRE追加 実装計画

作成日: 2026-01-21
最終更新日: 2026-01-21

## 目的
KG-FITバックエンド（entity: テキスト埋め込み初期化 + seed階層正則化）に **PairRE** を追加し、既存の「追加前KG / 追加後KG（updated_triples）」を流用した before/after の再学習・再評価を、TransE（既存）と同等の手順で実行可能にする。

狙い:
- TransEよりも one-to-many / many-to-one を表現しやすい相互作用（interaction）へ置き換え、
  「追加トリプルの偏り（tailハブ化等）」に対するロバスト性が改善するかを検証できる状態にする。
- KG-FITの正則化（anchor/cohesion/separation）は **entity embedding のみに適用**する現行設計を維持し、
  relation表現・interactionだけを差し替え可能にする。

## スコープ
- 対象: `embedding_backend="kgfit"` のPyKEENモデル構築経路
- 追加対象: PairRE（まずは実数表現）
- 対象外（本計画ではやらない）:
  - relation側のテキスト埋め込み導入
  - KG-FITの正則化設計の変更（重みや定義の刷新）
  - PairRE以外（RotatE/ComplEx等）の同時導入

## 参照
- KG-FITバックエンド標準仕様: [docs/rules/RULE-20260119-TAKG_KGFIT-001.md](../rules/RULE-20260119-TAKG_KGFIT-001.md)
- 正則化高速化/運用標準: [docs/rules/RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001.md](../rules/RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001.md)
- KG-FITでの再評価（full pipeline）: [docs/records/REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001.md](REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001.md)
- UCB介入（KG-FIT）: [docs/records/REC-20260121-UCB_KGFIT_INTERVENTIONS-001.md](REC-20260121-UCB_KGFIT_INTERVENTIONS-001.md)

## 現状把握（前提）
- 現行のKG-FITバックエンドは `model="TransE"` に固定されている（他モデルはエラー）。
- entity representation は `KGFitEntityEmbedding` を使い、正則化（anchor/cohesion/separation）が `regularization_term` に加算される。
- relation representation は通常の `Embedding` であり、relationテキスト埋め込みは利用していない。

この状態から、**interaction と relation representation を PairRE 用に拡張**する。

## 実装方針（設計案）

### 方針A（第一候補）: ERModel + PairREInteraction を直接構築
既存のKG-FIT（TransE）と同じく、PyKEENの `ERModel` を直接構築する。

- entity representation: 既存の `KGFitEntityEmbedding` をそのまま利用（`unique=True` を維持）
- relation representation: PairREが要求する形へ合わせる
  - 典型的に PairRE は relation ごとに2つのベクトル（$r_h, r_t$）を持つ
  - 実装候補:
    1) `Embedding(num_relations, 2*dim)` を持ち、forwardで `(r_h, r_t)` にsplitする薄いラッパ
    2) `ModuleList([Embedding(...), Embedding(...)])` で2本持つ
- interaction: PyKEENに `PairREInteraction` が提供されている場合はそれを使う
  - もし Interaction API が合わない場合は、PyKEENの PairRE実装（model/interaction）を参照して
    「ERModel向けに呼べる最小互換のInteraction」を自前実装する

メリット:
- KG-FIT（entity側フック/正則化）の設計を壊さない
- 既存の `train_model()` / `score_triples()` / `evaluate()` のI/Fに乗せやすい


### 方針B（代替）: PyKEEN標準 PairRE model を使い、entity representation を差し替える
PyKEENの `PairRE` model をそのまま `pipeline(model="PairRE", ...)` で作り、
entity representation だけKG-FITに差し替える。

リスク:
- 標準model内部で representation の組み立てが固定されている場合、差し替えが難しい
- 既存のKG-FIT「正則化を埋め込み層から注入する」設計と噛み合わない可能性

→ まずは方針Aで進め、Bは詰まった場合の退避にする。

## 変更点（予定）

### コード
- `simple_active_refine/embedding.py`
  - `embedding_backend="kgfit"` のときに `model` を `TransE | PairRE` で分岐できるようにする
  - PairRE用の interaction / relation representation を組み立てる
- （必要なら新規）`simple_active_refine/kgfit_pairre_representation.py`（または既存ファイルへ追記）
  - PairRE用 relation representation（2本のrelation embedding）を提供

補足:
- `simple_active_refine/kgfit_regularizer.py` / `kgfit_representation.py` は原則変更しない（entity側のみのまま）。

### 設定ファイル
- 新規のサンプル設定を追加（例）
  - `config_embeddings_kgfit_pairre_fb15k237.json`
  - 既存の `config_embeddings_kgfit_fb15k237.json` と並べて、modelだけ差し替え可能にする

### CLI/実験運用
- `scripts/train_initial_kge.py` は既存I/Fのまま利用（embedding configで `model` を切替）
- `retrain_and_evaluate_after_arm_run.py` / `run_full_arm_pipeline.py` は構造変更なしで走る状態を目標

## 受け入れ基準（Acceptance Criteria）

### 機能要件
- `embedding_backend="kgfit"` かつ `model="PairRE"` が学習を完走できる
- before/after の再学習・評価が既存の手順で実行でき、`summary.json` が生成される
- 既存の `model="TransE"`（KG-FIT）の挙動を壊さない

### 品質/再現性
- `.cache/kgfit/` 成果物参照、neighbor_k、正則化重みなどは既存仕様に従う
- 同一seedでの学習が再現可能（少なくとも手元検証で致命的な不安定がない）

### 最低限のテスト
- PairRE用 relation representation のshape/forwardが期待通りであるユニットテストを追加
- KG-FIT backend のモデル構築が `TransE` と `PairRE` の両方で通る簡易テストを追加

## 実装ステップ

### Step 0: PyKEEN側API確認（ブロッカー潰し）
- この環境のPyKEENに以下が存在するか確認
  - `PairREInteraction` の有無と forward シグネチャ
  - PairREが期待する relation representation の型/shape
- 期待shapeが分かったら、relation representation の最小実装を確定

### Step 1: PairRE用 relation representation 実装
- (案1) 1本のEmbeddingで `2*dim` を持ち、forwardで `(r_h, r_t)` を返す
- 返す型/shapeは、Interaction が受け付ける形に合わせる

### Step 2: `embedding.py` のKG-FIT backend 拡張
- `model` のガードを `TransE | PairRE` に緩和
- PairRE分岐を追加し、ERModelを構築
- 既存の学習・評価・保存・ロードの経路が動くように調整

### Step 3: 設定ファイル追加
- `config_embeddings_kgfit_*pairre*.json` を追加
- 既存のKG-FIT設定と差分（model名、必要なら interaction kwargs）を最小化

### Step 4: ユニットテスト追加
- `tests/` に PairRE用 representation のテストを追加
- `pytest -q` で最低限の回帰を確認

### Step 5: スモーク実行
- 小さいepoch（例: 2〜5）で `scripts/train_initial_kge.py` が完走することを確認
- ついでに `score_triples()` がエラー無く動くことを確認

### Step 6: 本番相当の比較実験
- `test_data_for_nationality_v3` を対象に
  - before KG: baseline train/valid/test
  - after KG: 既存 run の `updated_triples`（追加後KG）
 で、TransE(KG-FIT) vs PairRE(KG-FIT) を同条件で比較

## リスクと対策
- PairREのInteractionが要求する relation representation のshapeが想定と異なる
  - 対策: Step 0でPyKEEN実装を確認し、必要なら最小Interactionを自前実装
- PairREが不安定（発散、学習が進まない）
  - 対策: 学習率/正則化重み/embedding_dimを小さくしてスモーク→段階的に戻す
- 追加データの偏り問題（tailハブ化）が interaction だけでは改善しない
  - 対策: 既存の介入（tail-cap等）と「PairRE導入」を交差させ、要因分解する

## 成果物
- 実装: KG-FIT backend の PairRE 対応
- 設定: PairRE用 KG-FIT embedding config
- テスト: PairRE relation representation のユニットテスト
- 実験結果: before/after の比較サマリ（summary.json、必要ならrecord追記）
