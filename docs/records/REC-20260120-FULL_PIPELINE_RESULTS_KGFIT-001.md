# REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001: REC-20260118 の再評価（KGE=KG-FIT）

作成日: 2026-01-20  
最終更新日: 2026-02-01

参照:
- [REC-20260118-FULL_PIPELINE_RESULTS-001](REC-20260118-FULL_PIPELINE_RESULTS-001.md): 再実験対象（TransE）
- [REC-20260119-TAKG_KGFIT-001](REC-20260119-TAKG_KGFIT-001.md): TAKG移行（KG-FIT採用）に向けた目的/背景
- [RULE-20260119-TAKG_KGFIT-001](../rules/RULE-20260119-TAKG_KGFIT-001.md): KG-FITバックエンドの標準仕様

---

## 0. 目的

[REC-20260118-FULL_PIPELINE_RESULTS-001](REC-20260118-FULL_PIPELINE_RESULTS-001.md) で観測された「トリプル追加後に target score / Hits@k が悪化し得る」現象に対して、KGE を **KG-FIT（テキスト属性 + seed階層正則化）**に置き換えた場合に、

- target triple の平均スコア（target score）の悪化が緩和されるか
- Hits@k / MRR の悪化が緩和されるか

を **同一の追加前/追加後KG（既に作成済み）**を使って比較する。

背景（狙いの根拠）:
- [REC-20260119-TAKG_KGFIT-001](REC-20260119-TAKG_KGFIT-001.md) で整理した通り、構造のみKGEでは追加トリプルにより局所的に過制約になりやすい。
- KG-FITは、固定テキスト埋め込みと seed 階層（クラスタ）由来の正則化（anchor/cohesion/separation）をKGE学習へ加算し、埋め込みの自由度を増やすことでこの悪化を緩和することを期待する。

---

## 1. 再実験の方法

### 1.1 対象（追加前/追加後KG）

追加前/追加後KGは、[REC-20260118-FULL_PIPELINE_RESULTS-001](REC-20260118-FULL_PIPELINE_RESULTS-001.md) で作成済みのものをそのまま利用した。

- 共通 dataset_dir（追加前KG）: `/app/experiments/test_data_for_nationality_v3`
- target relation: `/people/person/nationality`
- target_triples: `/app/experiments/test_data_for_nationality_v3/target_triples.txt`（117）

追加後KG（runごとに異なる updated_triples）:
- exp_B4 rerun（LLM-policy / priors無し）
  - run_dir: `/app/experiments/20260118/exp_B4_llm_25_rerun_20260118b/arm_run`
  - updated_triples_dir: `<run_dir>/retrain_eval/updated_triples`
  - 追加トリプル（used）: 11195
- arm10 + priors（LLM-policy）
  - run_dir: `/app/experiments/20260118/exp_B4_llm_25_arm10_priors_20260118a/arm_run`
  - updated_triples_dir: `<run_dir>/retrain_eval/updated_triples`
  - 追加トリプル（used）: 11747
- arm10 + priors（UCB）
  - run_dir: `/app/experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run`
  - updated_triples_dir: `<run_dir>/retrain_eval/updated_triples`
  - 追加トリプル（used）: 10185

### 1.2 KGE（KG-FIT）設定

- embedding config: `/app/config_embeddings_kgfit_fb15k237.json`
  - embedding_backend: `kgfit`
  - base model: `transe`
  - reshape_strategy: `full`（name/desc結合）
  - seed hierarchy / neighbor_k / regularizer weights: config記載の通り
- epochs: 100

### 1.3 評価方法

- 評価コード: `simple_active_refine.evaluation.IterationEvaluator`
  - target score: `KnowledgeGraphEmbedding.score_triples()` の平均（raw score）
  - Hits@1/3/10, MRR: `KnowledgeGraphEmbedding.evaluate()`（test.txt）

注意:
- KG-FITとTransEはスコアスケール（raw）が異なるため、**target scoreの絶対値は比較しない**（比較は Δ に寄せる）。
- 本再実験のKG-FIT実行では、target_triples 117件のうち **11件が unknown entity/relation として score 計算から除外**された（`KnowledgeGraphEmbedding.score_triples()` の仕様）。よって target score は scorable subset（106件）の平均である。

### 1.4 実行スクリプトと成果物

- 実行スクリプト: `tmp/debug/rerun_full_pipeline_results_20260118_with_kgfit.py`
- 実行ログ: `tmp/debug/rerun_full_pipeline_results_20260118_with_kgfit_ep100.log`

学習済みモデル/評価:
- before model（共通）: `/app/models/20260120/fb15k237_kgfit_transe_nationality_before_ep100_fullpipeline_001`
- after model / 評価（runごと）:
  - `<run_dir>/retrain_eval_kgfit_ep100_full_pipeline_results_kgfit_001/`
    - `summary.json`
    - `evaluation/iteration_metrics.json`
    - `evaluation/iteration_evaluation.md`

---

## 2. 結果

### 2.1 KG-FIT（本再実験）の指標

共通（before, KG-FIT）:
- Hits@1/3/10/MRR = 0.1127 / 0.2721 / 0.5882 / 0.2422
- target score = -2536.9576

| 条件 | n_added | target score（before→after, Δ） | Hits@1（Δ） | Hits@3（Δ） | Hits@10（Δ） | MRR（Δ） |
|---|---:|---:|---:|---:|---:|---:|
| LLM-policy（rerun） | 11195 | -2536.9576 → -2589.0773 (Δ=-52.1197) | +0.0196 | -0.0270 | -0.0490 | +0.0098 |
| LLM-policy（arm10+priors） | 11747 | -2536.9576 → -2573.8884 (Δ=-36.9309) | +0.0049 | -0.0515 | -0.0588 | -0.0126 |
| UCB（arm10+priors） | 10185 | -2536.9576 → -2540.3123 (Δ=-3.3548) | +0.0294 | -0.0172 | -0.0637 | +0.0144 |

### 2.2 参考: TransE（元実験）の指標

同一の追加後KG（updated_triples）に対して、元実験（TransE）では以下（抜粋）。
詳細は [REC-20260118-FULL_PIPELINE_RESULTS-001](REC-20260118-FULL_PIPELINE_RESULTS-001.md) を参照。

- LLM-policy（rerun）: target score Δ=-3.5748, MRR Δ=+0.0065
- LLM-policy（arm10+priors）: target score Δ=-3.3085, MRR Δ=-0.0026
- UCB（arm10+priors）: target score Δ=-0.1763, MRR Δ=+0.0080

---

## 3. 考察

### 3.1 「target score 悪化の緩和」について

- KG-FITでも、LLM-policy 2条件は target score が明確に悪化（Δは負）した。
- 一方で、UCB（arm10+priors）は target score の悪化が小さく、元実験（TransE）と同様に「3条件の中では最も安定」だった。

この結果だけを見ると、少なくとも本設定（`config_embeddings_kgfit_fb15k237.json`、epoch=100）では、
- **LLM-policy条件で起きる悪化をKG-FITが一貫して抑える**、とは言いにくい。

### 3.2 Hits/MRRの挙動

- KG-FITでは、3条件すべてで Hits@10 が低下した（Δ<0）。
- Hits@1 と MRR は、rerun と UCB で上昇したが、arm10+priors（LLM-policy）では MRR が低下した。

トップ（Hits@1/MRR）だけ改善しつつ全体のランキング（Hits@10）が崩れる可能性があり、
「追加トリプルが良い方向に効いた部分」と「ノイズ/過制約として効いた部分」が混在している可能性がある。

### 3.3 追加で確認したい点（次の打ち手）

- KG-FIT正則化重み（anchor/cohesion/separation）や neighbor_k をスイープし、
  - target score の悪化緩和
  - Hits@10 の低下抑制
  の両立点があるかを見る。
- baseline TransE と比較して KG-FIT の before 指標が低いため、
  - 学習率/バッチサイズ/epoch
  - embedding_dim（full=3072のままが妥当か、sliceで軽量化するか）
  を調整し、まず before 側の精度を揃えた上で再比較する。

---

## 結論（観測→含意→次アクション）

### 観測

- 同一の追加前/追加後KGを用いた再評価の範囲では、KG-FITでも LLM-policy 条件の target score 悪化（Δ<0）が観測された。
- 3条件の中では、UCB（arm10+priors）が target score の悪化が最も小さく「安定」だった。
- Hits@10 は3条件すべてで低下し、Hits@1/MRRは一部で改善するなど、指標間のトレードオフが見られた。

### 含意

- KG-FITの導入だけで「追加トリプルによる悪化」を一貫して抑えられるとは言えず、設定（正則化・近傍・学習条件）に依存する可能性が高い。
- target最適化（score）と全体ランキング（Hits/MRR）の同時最適化は自明ではなく、目的関数（$
\mathcal{L}=\zeta_1\mathcal{L}_{hier}+\zeta_2\mathcal{L}_{anc}+\zeta_3\mathcal{L}_{link}$
）の重み設計が効いている可能性がある。

### 次アクション

- KG-FITの正則化重み（anchor/cohesion/separation）と neighbor_k をスイープし、target score 悪化の緩和と Hits@10 低下抑制の両立点を探索する。
- baseline（TransE）と比較可能になるよう、before側の学習条件（lr/bs/epoch/embedding_dim）を調整して精度を揃えた上で再比較する。

---

## 更新履歴

- 2026-01-20: 新規作成。REC-20260118 の追加前/追加後KGを流用して KG-FIT で再学習・評価した結果を記録。
