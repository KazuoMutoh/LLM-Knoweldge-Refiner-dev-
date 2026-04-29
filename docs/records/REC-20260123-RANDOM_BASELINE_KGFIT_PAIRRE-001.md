# REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001: KG-FIT(PairRE) arm=10/UCB の改善がランダム追加（train_removedサンプリング）より良いか比較する実験計画

作成日: 2026-01-23
最終更新日: 2026-02-01

## 目的

[docs/records/REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001.md](REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001.md) の結果より、KG-FIT(PairRE) + arm=10 + selector=UCB + priors=off が target score 改善に寄与する可能性が高いことが示唆された。

本計画では、**同じ候補集合（train_removed.txt）から同数のトリプルをランダムに追加**した場合と比較して、
UCB（arm-runで選ばれた追加）の方が良いかどうかを検証する。

- 比較対象: **KG-FIT(PairRE)** 条件のみ（TransEは今回は扱わない）
- 追加元: `train_removed.txt`（候補トリプル）
- 比較軸: target triple の再スコア（minmax(train)）を主、ランキング指標（Hits/MRR）は副

## 背景・参照

- 参照実験（UCB / priors=off / arm=10 / ep=100）
  - run_dir: `/app/experiments/20260122/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_20260122b`
  - summary.json: `/app/experiments/20260122/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_20260122b/arm_run/retrain_eval/summary.json`
  - UCBで最終的に追加されたトリプル数（集約・重複除去後）:
    - `n_added_used = 10232`

## 仮説

- H1: UCBにより選ばれた追加トリプル（= 参照実験の accepted_added_triples 集約）で再学習した after モデルは、
  `train_removed.txt` から同数ランダムに追加した場合よりも、target triples の minmax(train) スコアΔが大きい（平均Δが高い）。
- H2: ランキング指標（MRR/Hits）は、UCB優位が必ずしも成立しない（target最適化とトレードオフの可能性がある）。

## 実験条件（固定）

参照実験と揃える。

- Dataset（学習に使うdir_triples）: `/app/experiments/test_data_for_nationality_v3_kgfit`
- Target relation: `/people/person/nationality`
- Target triples: `/app/experiments/test_data_for_nationality_v3/target_triples.txt`
- Candidate triples（ランダム追加の母集団）: `/app/experiments/test_data_for_nationality_v3/train_removed.txt`
- before model: `/app/models/20260122/fb15k237_kgfit_pairre_nationality_v3_before_ep100`
- embedding config: `/app/config_embeddings_kgfit_pairre_fb15k237.json`
- num_epochs: 100
- priors: off（ランダム追加実験では arm pipeline を走らせないため影響無しだが、参照実験と軸を揃える意図で明記）

## 比較条件

### 条件U（参照）: UCB（arm=10, priors=off）

- 参照runの after model と出力をそのまま用いる。

### 条件R（新規）: Random add（train_removed から同数をサンプリングして追加）

- `train_removed.txt` から、**UCBと同数 N=10232** を **without replacement** でランダムサンプリングして train に追加する。
- サンプリングの乱数seedを変えて複数回繰り返し、ランダムの分散を推定する。

推奨反復数:
- まずは `n_seeds=5`（seed: 0..4）
- 可能なら `n_seeds=10` まで増やす（計算時間と相談）

注意（公正性・混入要因）:
- ランキング指標やtarget再スコアの比較は、
  - usable target triples 数（unknownの除外数）
  - afterモデルの train entity/relation の変化
  の影響を受け得る。
- よって各ランのレポートに「target coverage（before/afterでknownなtarget数）」を必ず併記する。

## 実行手順（ランダム追加1回分）

### 1) サンプル作成

入力:
- `train_removed.txt`
- N=10232
- seed = 任意

出力:
- `accepted_added_triples.tsv`（3列TSV: head\trelation\ttail）

本プロジェクトの後段（再学習/評価）を流用するため、arm-run互換の形で run_dir を構成する:

- `<RUN_DIR>/arm_run/iter_1/accepted_added_triples.tsv`

（補足）`retrain_and_evaluate_after_arm_run.py` は `iter_*/accepted_added_triples.tsv` を優先集約する。

### 2) 再学習・評価

`retrain_and_evaluate_after_arm_run.py` を用いて、train ∪ sampled_triples の updated_triples を作り、after を再学習して評価する。

- 出力:
  - `<RUN_DIR>/arm_run/retrain_eval/summary.json`
  - `<RUN_DIR>/arm_run/retrain_eval/model_after/`
  - `<RUN_DIR>/arm_run/retrain_eval/updated_triples/`（train_after, added_triples.tsv など）

## 実行コマンド例（ランダム seed=0）

※実行は本計画のGo合意後。

```bash
RUN_DIR=/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed0_20260123a
ARM_RUN_DIR="$RUN_DIR/arm_run"

# 1) accepted_added_triples.tsv の作成（例: pythonスクリプトで生成）
#  - 出力先: "$ARM_RUN_DIR/iter_1/accepted_added_triples.tsv"

# 2) 再学習・評価
python3 /app/retrain_and_evaluate_after_arm_run.py \
  --run_dir "$ARM_RUN_DIR" \
  --dataset_dir /app/experiments/test_data_for_nationality_v3_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_v3/target_triples.txt \
  --model_before_dir /app/models/20260122/fb15k237_kgfit_pairre_nationality_v3_before_ep100 \
  --after_mode retrain \
  --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
  --num_epochs 100 \
  2>&1 | tee "$RUN_DIR/run.log"
```

## 評価・比較方法

### 1) summary.json 指標（各seedで収集）

- `target_score_change`（raw; ただしモデル間スケール差が大きいので主指標にはしない）
- `mrr_change`, `hits_at_k_change`
- `n_triples_added`（想定=10232）

### 2) target triples 再スコア（主指標）

参照実験で用いた「minmax(train) による target score Δ」比較を、
Random条件（各seed）にも適用して比較する。

- 参照レポート: `/app/experiments/20260122/normalized_target_score_comparison_20260122b.md`
- 生成スクリプト（暫定）: `/app/tmp/debug/normalize_target_scores_20260122b.py`

運用案:
- 上記スクリプトを複製して、
  - PairRE before は固定
  - PairRE after を各seedの `model_after` に差し替え
  - UCB after との比較表（Δmean, improved_frac など）を出す

### 3) 統計的比較（簡易）

- Random（seed反復）の `Δmean(minmax(train))` 分布を作り、
  - UCBのΔmeanがどのパーセンタイルに位置するか（例: 90%以上なら「ランダムより良い」根拠）
  - 95% CI（ブートストラップ or seed分散）
  を提示する。

## 成果物（保存場所）

- 実行ディレクトリ: `/app/experiments/20260123/`
- 各seedの run_dir:
  - `exp_random_add10232_priors_off_kgfit_pairre_ep100_seed{seed}_20260123a`
- 収集物:
  - `arm_run/retrain_eval/summary.json`
  - `arm_run/retrain_eval/model_after/`
  - `arm_run/retrain_eval/updated_triples/added_triples.tsv`（実際に追加に使われた集合）
  - （任意）UCB vs Random の比較Markdown（20260123版）

## 実行結果（暫定: Random seed 0..3）

この節は、2026-01-23時点で完了している seed=0..3 の結果に基づく暫定まとめ（seed=4は当時進行中）である。

### 実験runと成果物

- UCB（参照）:
  - run_dir: `/app/experiments/20260122/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_20260122b`
  - summary.json: `/app/experiments/20260122/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_20260122b/arm_run/retrain_eval/summary.json`
- Random（新規; N=10232）:
  - seed=0: `/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed0_20260123a`
  - seed=1: `/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed1_20260123a`
  - seed=2: `/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed2_20260123a`
  - seed=3: `/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed3_20260123a`

比較レポート（summary.json + minmax(train) 追記済み）:

- `../../experiments/20260123/compare_ucb_vs_random_seed0_3_summary_20260123a.md`
- （minmax(train) セクション単体）`../../experiments/20260123/minmax_target_rescore_section_seed0_3.md`

使用スクリプト:

- `../../scripts/sample_random_triples.py`（train_removed.txt から非復元抽出）
- `../../scripts/run_random_baseline_batch.py`（seed=0..4 バッチ実行）
- `../../scripts/compute_minmax_rescore_section.py`（minmax(train) 再スコア比較セクション生成）

### 主要結果（seed0..3）

#### 1) summary.json 指標（raw）

UCBは `target_score_change(raw)` が正で、Random(seed0..3) は全て負。
ただし raw スコア差はモデル間スケールの影響が大きいので、主張は次項（minmax(train)）を主とする。

#### 2) minmax(train) 正規化 target 再スコア（主指標）

PairRE-before を固定し、各PairRE-after（UCB/Random）で target triples を再スコア。
各afterモデルの `train` に対して min-max 正規化（0..1）した上で、Δ（after-before）を算出。

- usable targets（全モデルで known な target の共通集合）: 106 / 117（11件は除外）
- UCB:
  - Δmean = +0.041186
  - improved_frac = 0.924528
- Random(seed0..3):
  - Δmean は全seedで負（例: seed2 = -0.170786）
  - improved_frac は全seedで 0
  - Δmean aggregate: mean = -0.153068, std = 0.0160266

結論（暫定）:

- seed0..3 の範囲では、**UCB（arm=10）で選ばれた追加は、同数ランダム追加よりも minmax(train) による target 改善が明確に大きい**。

## 考察（暫定）

- ランダム追加は target に対して一貫して悪化方向（minmax(train) Δmean < 0）であり、`train_removed` からの「無差別追加」は target 最適化に不利な可能性が高い。
- 一方で、ranking 指標（MRR/Hits）は seed により挙動が混在し、target 指標と一致しない（トレードオフ/ノイズの可能性）。

## 失敗時・リスク

- `train_removed.txt` の行数が N 未満の場合: N を減らす（ただし今回のN=10232は成立している前提）
- after再学習で時間が掛かる: まず seed を少数（5）に絞って比較し、必要なら追加実行
- target coverage が seed により変動する場合:
  - coverageを必ず明記し、coverageが揃うサブセットでの比較（共通集合）も併記する

## 次のTODO

- [x] Random追加用のサンプリングスクリプト（TSV出力、seed指定、重複排除）を `scripts/` に用意（`scripts/sample_random_triples.py`）
- [x] seed=0..3 の実行と summary.json 収集
- [x] seed=0..3 の UCB vs Random 比較レポート作成（minmax(train) セクション追記含む）
- [ ] seed=4 が完了していれば、seed0..4 版の比較レポートに拡張

---

## 結論（観測→含意→次アクション）

### 観測

- seed0..3 の範囲では、UCB（arm=10/priors=off）で選ばれた追加は、同数ランダム追加より minmax(train) 正規化の target 再スコア改善が明確に大きかった。
- ランキング指標（MRR/Hits）は seed により挙動が混在し、target 指標と一致しないケースがある。

### 含意

- train_removed からの無差別追加は target 最適化に不利であり、選別（UCBなど）が有効に働き得る。
- ただし「target改善」を主目的に据える場合でも、ランキング指標の劣化が許容できるか（もしくは抑制できるか）を別途評価する必要がある。

### 次アクション

- seed=4（可能なら追加seed）を完走させ、Random分布の推定精度を上げる。
- UCB/Randomの比較は [RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001](../rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md) に沿って、usable targets数（共通集合）と正規化方法（minmax(train)）を必ず併記する。
