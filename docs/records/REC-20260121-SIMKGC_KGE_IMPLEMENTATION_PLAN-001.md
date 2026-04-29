# REC-20260121-SIMKGC_KGE_IMPLEMENTATION_PLAN-001: SimKGCをKGEバックエンドとして追加する実装計画

作成日: 2026-01-21
最終更新日: 2026-01-21

## 目的
本プロジェクトのKGE（`simple_active_refine/embedding.py` の `KnowledgeGraphEmbedding`）に、SimKGC（ACL 2022）相当の **テキストベースKGCモデル**を追加し、既存の実験パイプライン（学習→スコアリング→評価→ARM反復精錬）で **PyKEEN系（TransE/PairRE/KG-FIT）と並列比較**できる状態にする。

## 参照資料
- 論文（ローカル）: [docs/external/sim-kgc-論文.pdf](../external/sim-kgc-%E8%AB%96%E6%96%87.pdf)
  - 主要要素: bi-encoder、InfoNCE + additive margin、3種のnegative（in-batch / pre-batch / self-negative）、filtered ranking評価、（任意）グラフ近傍によるrerank
- GitHub（公式実装）: https://github.com/intfloat/SimKGC/tree/main?tab=readme-ov-file
  - 重要ファイル: `models.py`, `trainer.py`, `evaluate.py`, `preprocess.py`, `doc.py`

## 現状（本リポジトリ側の制約）
- 既存KGEはPyKEEN前提（`embedding_backend in {pykeen, kgfit}`）。SimKGCはPyKEEN互換ではない。
- 既存データディレクトリには、テキスト属性のTSVが存在することが多い（例: `data/FB15k-237/{entity2text.txt, entity2textlong.txt, relation2text.txt}`）。
- 依存関係として `transformers` が未導入（現環境では `import transformers` が失敗）。SimKGC導入時は追加が必要。

## SimKGCの要点（論文・実装から）
### モデル
- bi-encoder: 
  - クエリ側: $(h, r)$ をテキスト化してBERTでエンコード（`hr_bert`）
  - 候補側: entity $t$ をテキスト化してBERTでエンコード（`tail_bert`）
- スコア: $\phi(h,r,t)=\cos(e_{hr}, e_t)\in[-1,1]$（実装は正規化ベクトルの内積）

### 学習
- 損失: InfoNCE + additive margin $\gamma$（正例logitから引く）
- negatives:
  - in-batch: 同一バッチ内の他のtail
  - pre-batch: 過去バッチのtail埋め込みをキューに保持
  - self-negative: head entity自体をhard negativeとして追加
- 温度 $\tau$ は学習可能にできる（実装は `log_inv_t`）

### 評価
- filtered setting（known true triplesをmaskしてrank計算）
- forward / backward の平均（(h,r,?) と (t, r^{-1}, ?)）
- （任意）graph-based reranking: headのk-hop近傍に加点（neighbor_weight, rerank_n_hop）

## 実装方針（このリポジトリでの統合設計）
### 方針A（推奨）: `embedding_backend="simkgc"` を追加し、内部に専用実装を持つ
- 既存の `KnowledgeGraphEmbedding.train_model()` を拡張し、`embedding_backend == "simkgc"` の場合は **SimKGC用Trainer/Model/Evaluator** に分岐
- 既存パイプライン呼び出し（`scripts/train_initial_kge.py` や `retrain_and_evaluate_after_arm_run.py` 等）を保ったまま、設定だけで切替可能にする

### 方針B（代替）: SimKGC専用クラスを新設し、呼び出し側で分岐
- 変更箇所が増えやすい（既存の `KnowledgeGraphEmbedding` の利用箇所が多い）

結論: 方針Aで進める（I/F互換を優先）。

## 追加するコンポーネント（ファイル案）
### 1) SimKGCコア実装（新規）
- `simple_active_refine/simkgc/model.py`
  - `SimKGCModel`（BERT bi-encoder、pooling、温度、pre-batch queue、self-negative）
- `simple_active_refine/simkgc/data.py`
  - データ構造（Example）、tokenize、collate、triplet mask（false negatives除外）
- `simple_active_refine/simkgc/trainer.py`
  - 1 epoch学習、mixed precision、DataParallel（必要なら）
- `simple_active_refine/simkgc/evaluate.py`
  - entity埋め込み事前計算、hr埋め込み計算、全entityとの内積、filtered mask、metrics（MRR/Hits@k）

※ GitHub実装の概念を踏襲しつつ、コードは本リポジトリの規約（loggingやI/F）に合わせて再実装する（直接コピーは避ける）。

### 2) `KnowledgeGraphEmbedding` への統合（既存修正）
- `simple_active_refine/embedding.py`
  - `train_model()` に `embedding_backend == "simkgc"` 分岐
  - `__init__()` ロードにもSimKGCのmodel artifactを扱えるよう拡張（`trained_model.pkl`ではなく、`simkgc_checkpoint.pt`等）
  - `score_triples()` / `evaluate()` をSimKGC実装に委譲

### 3) データ前処理（新規スクリプト）
- `scripts/prepare_simkgc_dataset.py`
  - 入力: `dir_triples/`（`train.txt`, `valid.txt`, `test.txt`, `entity2text*.txt`, `relation2text.txt`）
  - 出力（SimKGC形式）:
    - `train.txt.json`, `valid.txt.json`, `test.txt.json`（list[dict]）
      - 例: `{head_id, head, relation, tail_id, tail}`
    - `entities.json`（list[dict]）
      - 例: `{entity_id, entity, entity_desc}`
  - relation text は `relation2text.txt` を優先利用（無い場合は論文/実装相当の正規化をfallback）
  - sanity check: 同一surface formに複数relationが落ちないこと（SimKGC実装が厳密）

### 4) 学習スクリプト（新規 or 既存拡張）
- 既存の `scripts/train_initial_kge.py` をそのまま使うことを目標
  - `embedding_config` 内で `embedding_backend: "simkgc"` を指定
  - `model` は無視し、`simkgc` セクションでハイパラを渡す
- 追加オプション例（embedding_configの提案）:
  ```json
  {
    "embedding_backend": "simkgc",
    "simkgc": {
      "pretrained_model": "bert-base-uncased",
      "pooling": "mean",
      "max_num_tokens": 50,
      "batch_size": 256,
      "epochs": 10,
      "lr": 1e-5,
      "weight_decay": 1e-4,
      "use_amp": true,
      "use_link_graph": false,
      "use_self_negative": true,
      "pre_batch": 2,
      "pre_batch_weight": 0.5,
      "additive_margin": 0.02,
      "t_init": 0.05,
      "finetune_t": true
    }
  }
  ```

## モデル保存・読み込み（artifact設計）
### 保存先（既存と合わせる）
- `dir_save/`（例: `models/202601xx/...`）配下に以下を保存
  - `simkgc_checkpoint.pt`（model state + args）
  - `simkgc_metrics.json`（学習/検証のログ、最良epoch等）
  - `simkgc_entity_embeddings.pt`（評価高速化用のentity tensor cache；任意）
  - `training_triples/` はSimKGCでは不要だが、互換のため `entities.json` 等を同梱しても良い

### 読み込み
- `KnowledgeGraphEmbedding(model_dir=...)` でロードできること（既存のパイプラインが期待）
- `model_dir` 内のファイル有無でPyKEENモデルかSimKGCモデルかを判別

## スコアリング互換（0-1正規化）
- SimKGCは $\cos\in[-1,1]$ を返すため、既存の「0-1スコア前提」へ合わせる必要がある
- 推奨:
  - `raw_score = cosine` を保持
  - `score = (raw_score + 1) / 2` をデフォルト
  - 既存と同様に train score 分布でmin-max正規化するオプションも用意（`normalization` の選択肢として追加）

## 評価の統合（Hits/MRR）
- SimKGC標準のrank評価（filtered setting）を実装し、`KnowledgeGraphEmbedding.evaluate()` から呼べるようにする
- 本リポジトリ既存の評価と差異が出やすい点:
  - inverse triples の扱い（SimKGCは評価時に forward/backward 両方向を平均）
  - filtered mask の参照範囲（train/valid/testをまとめてknownとしてmask）
  - rerank（近傍加点）のON/OFF

→ 実装初期は rerank をOFFにし、まずは再現性の高い基本評価を揃える。

## 依存関係・環境
- 必須:
  - `transformers>=4.15`
  - （多くの場合）`sentencepiece`（モデルによっては必要）
- 任意（高速化）:
  - `faiss-cpu` / `faiss-gpu`（entity埋め込みのANN検索）

`requirements.txt` に追加し、dev containerで再ビルド/インストールする。

## テスト計画
- `tests/test_simkgc_smoke.py`（新規）
  - 超小規模データ（`tests/resources/` か `tmp/` 生成）で1 epoch学習 → evaluate が走ること
  - `score_triples()` が shape / 範囲（0-1）を満たすこと
- 既存の `scripts/train_initial_kge.py` に対し、`embedding_backend=simkgc` のconfigで起動できること

## 実験計画（最小再現→統合）
1. データ変換のみ: `data/FB15k-237` を `*.txt.json`/`entities.json` に変換し、sanity check
2. 学習スモーク: batch_size小（例: 16〜64）、epochs=1、GPU/CPU両対応で動作確認
3. 評価スモーク: testの数十件でMRR/Hitsが計算できること
4. 統合実験: `experiments/test_data_for_nationality_v3` 相当の小規模KGで before/after 評価導線に載せる

## 主要リスクと対策
- **メモリ**: SimKGCはnegativesを大量に使うためbatch_size依存が強い
  - 対策: まず小batchで動く構成をデフォルトにし、性能再現は別途GPUリソースがあるときに実施
- **filtered maskのコスト**: `(h,r)` ごとにknown tailsをmaskする処理が重い
  - 対策: known triplets辞書を事前構築、mask対象が巨大な場合はログと上限（サンプリング/skip）
- **テキスト資源の品質**: entity2textlong が空/短いと性能が落ちる
  - 対策: 最低限 name + desc を結合、欠損はfallback（nameのみ）
- **依存関係追加**: `transformers` 導入でimageサイズ/インストール時間が増える
  - 対策: pin最小限、必要ならextra requirements化

## 作業ステップ（チェックリスト）
1. `requirements.txt` に `transformers` 等を追加
2. `scripts/prepare_simkgc_dataset.py` を実装し、FB15k-237で動作確認
3. `simple_active_refine/simkgc/` にモデル/データ/学習/評価を実装
4. `simple_active_refine/embedding.py` に `embedding_backend="simkgc"` を統合
5. スモークテスト追加（1 epoch train + score + eval）
6. `run_full_arm_pipeline.py` / `retrain_and_evaluate_after_arm_run.py` のKGE再学習経路でSimKGCが選べることを確認

## トレーサビリティ（参照コード）
- 既存KGE I/F: [simple_active_refine/embedding.py](../../simple_active_refine/embedding.py)
- 既存テキスト資源読込: [simple_active_refine/knoweldge_retriever.py](../../simple_active_refine/knoweldge_retriever.py)
- SimKGC GitHub: https://github.com/intfloat/SimKGC/tree/main

