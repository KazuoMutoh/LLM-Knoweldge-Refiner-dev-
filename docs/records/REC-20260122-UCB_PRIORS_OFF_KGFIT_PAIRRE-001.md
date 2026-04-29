# REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001: arm=10 + UCB + priors=off で KGE(TransE vs KG-FIT(PairRE)) を比較する実験計画

作成日: 2026-01-22
最終更新日: 2026-02-01

更新概要（2026-01-23）:
- 本計画に基づく実験（20260122b）の実行結果（summary.json / 比較表 / target再スコア正規化）を追記。

更新概要（2026-02-01）:
- 結論を「観測→含意→次アクション」形式に統一。

## 運用メモ（このファイルの位置付け）

- 本ドキュメントは **実験計画書（実行手順の定義）** であり、この時点で実験を開始しない。
- 実行は、別途「実行開始の合意（Go）」を取ってから行う。

## 目的

- **target_triple の score 改善**に着目し、同一の arm-run 設定（arm=10, selector=UCB, priors=off）において、KGE を **TransE** と **KG-FIT(PairRE)** に切り替えたときの before/after を比較する。
- 特に、「priors の有無」「selector」「incident triples」などの要因と混ざらないように、**priors=off を強制**し、arm数も固定して比較する。

## 比較条件（2条件）

共通（固定）:
- Dataset: `/app/experiments/test_data_for_nationality_v3`
- Target relation: `/people/person/nationality`
- Target triples: `target_triples.txt`
- Candidate triples: `train_removed.txt`
- Rule生成〜arm生成〜arm-run〜after再学習/評価までを `run_full_arm_pipeline.py` で通し実行
- **arm=10 を固定**: `--n_rules 10` かつ `--k_pairs 0`（singleton armのみ）
- Selector: `--selector_strategy ucb`
- **priors=off を強制**: `--disable_relation_priors`
- Epoch: `--num_epochs 100`

条件A（TransE）:
- `--embedding_config /app/config_embeddings.json`

条件B（KG-FIT(PairRE)）:
- `--embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json`

補足（重要）:
- `run_full_arm_pipeline.py` の `--model_dir` は初期ルールプール生成（KGEスコア計算）に使われるため、**TransE条件とKG-FIT(PairRE)条件で別々の before model を用意**して比較する。

## 重要な注意（priors=off の厳密化）

`ArmPipelineConfig` はデフォルトで `<dataset_dir>/relation_priors.json` を自動検出して priors を適用するため、
**priors=off を保証するには `--disable_relation_priors` を必ず付ける**。

## 前提（KG-FIT 実行に必要な成果物の用意）

`test_data_for_nationality_v3` は最小構成の triples のみなので、KG-FIT(PairRE) では以下が必要:
- `entity2text.txt` / `entity2textlong.txt` / `entities.txt` / `relation2text.txt`
- `.cache/kgfit/`（事前計算済み埋め込み + seed階層 + neighbor clusters）

本リポジトリでは、FB15k-237 のフルデータセット側にこれらが存在する:
- `/app/data/FB15k-237/entity2text*.txt`
- `/app/data/FB15k-237/entities.txt`
- `/app/data/FB15k-237/relation2text.txt`
- `/app/data/FB15k-237/.cache/kgfit/`

安全のため、KG-FIT用に dataset_dir を複製してから必要ファイルをコピーする。

```bash
BASE_DS=/app/experiments/test_data_for_nationality_v3
KGFIT_DS=/app/experiments/test_data_for_nationality_v3_kgfit

rm -rf "$KGFIT_DS" && mkdir -p "$KGFIT_DS"
cp -a "$BASE_DS"/* "$KGFIT_DS"/

cp -a /app/data/FB15k-237/entity2text.txt /app/data/FB15k-237/entity2textlong.txt \
  /app/data/FB15k-237/entities.txt /app/data/FB15k-237/relation2text.txt \
  "$KGFIT_DS"/

mkdir -p "$KGFIT_DS/.cache"
cp -a /app/data/FB15k-237/.cache/kgfit "$KGFIT_DS/.cache/"
```

## 実行コマンド（推奨）

### 事前: KG-FIT(PairRE) before model の学習（ep=100）

KG-FIT(PairRE) 条件では、事前に before model を学習して `--model_dir` として指定する。

```bash
PAIRRE_BEFORE_DIR=/app/models/20260122/fb15k237_kgfit_pairre_nationality_v3_before_ep100

python3 /app/scripts/train_initial_kge.py \
  --dir_triples /app/experiments/test_data_for_nationality_v3_kgfit \
  --output_dir "$PAIRRE_BEFORE_DIR" \
  --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
  --num_epochs 100 \
  --skip_evaluation
```

### 条件A: TransE / arm=10 / UCB / priors=off / ep=100

```bash
RUN_DIR=/app/experiments/20260122/exp_ucb_arm10_priors_off_transe_ep100_20260122b

python3 run_full_arm_pipeline.py \
  --run_dir "$RUN_DIR" \
  --model_dir /app/models/20260116/fb15k237_transe_nationality \
  --target_relation /people/person/nationality \
  --dataset_dir /app/experiments/test_data_for_nationality_v3 \
  --target_triples /app/experiments/test_data_for_nationality_v3/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_nationality_v3/train_removed.txt \
  --n_rules 10 \
  --k_pairs 0 \
  --n_iter 25 \
  --k_sel 3 \
  --n_targets_per_arm 20 \
  --selector_strategy ucb \
  --disable_relation_priors \
  --after_mode retrain \
  --embedding_config /app/config_embeddings.json \
  --num_epochs 100 \
  2>&1 | tee "$RUN_DIR/run.log"
```

### 条件B: KG-FIT(PairRE) / arm=10 / UCB / priors=off / ep=100

```bash
RUN_DIR=/app/experiments/20260122/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_20260122b

python3 run_full_arm_pipeline.py \
  --run_dir "$RUN_DIR" \
  --model_dir /app/models/20260122/fb15k237_kgfit_pairre_nationality_v3_before_ep100 \
  --target_relation /people/person/nationality \
  --dataset_dir /app/experiments/test_data_for_nationality_v3_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_v3/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_nationality_v3/train_removed.txt \
  --n_rules 10 \
  --k_pairs 0 \
  --n_iter 25 \
  --k_sel 3 \
  --n_targets_per_arm 20 \
  --selector_strategy ucb \
  --disable_relation_priors \
  --after_mode retrain \
  --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
  --num_epochs 100 \
  2>&1 | tee "$RUN_DIR/run.log"
```

## 評価（比較の見方）

各RUNの以下を比較する:
- `arm_run/retrain_eval/summary.json`
  - target_triple score（mean, Δ）
  - Hits@k / MRR（before/after, Δ）

加えて、arm-run自体の挙動（evidence量・追加の偏り）を見る:
- `arm_run/iter_*/accepted_evidence.tsv`
- `arm_run/iter_*/accepted_added_triples.tsv`（incident込みの可能性あり）

## 期待される結果（仮説）

- KG-FIT(PairRE) が TransE より **target_triple score の改善が安定**（少なくとも劣らない）こと。
- priors=off でも UCB により一定の改善が出ること。

## 実行結果（20260122b）

### 実行した RUN

条件A（TransE）:
- run_dir: `/app/experiments/20260122/exp_ucb_arm10_priors_off_transe_ep100_20260122b`
- summary.json: `/app/experiments/20260122/exp_ucb_arm10_priors_off_transe_ep100_20260122b/arm_run/retrain_eval/summary.json`

条件B（KG-FIT(PairRE)）:
- run_dir: `/app/experiments/20260122/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_20260122b`
- summary.json: `/app/experiments/20260122/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_20260122b/arm_run/retrain_eval/summary.json`

### 集計レポート（自動生成）

- summary.json 比較表: [`/app/experiments/20260122/compare_arm10_ucb_priors_off_transe_vs_kgfit_pairre_20260122b.md`](../../experiments/20260122/compare_arm10_ucb_priors_off_transe_vs_kgfit_pairre_20260122b.md)
- target triples 再スコア + 正規化比較: [`/app/experiments/20260122/normalized_target_score_comparison_20260122b.md`](../../experiments/20260122/normalized_target_score_comparison_20260122b.md)

### 結果サマリ（重要点）

1) **target triple score（再スコア、minmax(train)）**
- target_triples は 117 本だが、unknown entity/relation を除いた共通集合で **106 本**を比較。
- TransE: minmax(train) の Δmean = **-0.0366**、improved_frac = **0.0**（全て悪化方向）
- KG-FIT(PairRE): minmax(train) の Δmean = **+0.0412**、improved_frac = **0.9245**（大半が改善方向）

2) **ランキング系指標（Hits/MRR; summary.json）**
- TransE: MRR Δ = **+0.00794**
- KG-FIT(PairRE): MRR Δ = **-0.0122**

3) 解釈（この計画の目的に対して）
- 本計画の主目的を「target_triple の score 改善」に置くなら、**KG-FIT(PairRE) の方が良さそう**（少なくとも今回の設定では、target再スコアで改善が支配的）。
- 一方で、標準的なリンク予測のランキング指標（Hits/MRR）では、今回 **TransE が改善し、KG-FIT(PairRE) は悪化**している点に注意。
- よって次の意思決定は「target score 最適化を最優先にする」か、「ランキング指標も同時に維持/改善したい」かで分岐する。

### 考察（REC-20260119-TAKG_KGFIT-001 を踏まえた解釈）

REC-20260119-TAKG_KGFIT-001 では、KG-FIT は「**Fine-tune KG with LLM**」として、
テキスト由来のグローバル意味（アンカー）と階層（クラスタ）制約を KGE の学習目的へ統合することで、
構造のみKGEで起きがちな「追加トリプルによる局所的な過制約」や「意味の逸脱」を緩和する、という位置づけだった。

今回の観測（20260122b）は、その設計意図と整合する面がある一方で、評価軸のズレも同時に示唆している:

1) **target score 改善（再スコア minmax(train)）が優勢だった理由（仮説）**
- target relation（`/people/person/nationality`）は、周辺のテキスト意味（国籍/人物/国）と整合しやすい。
- KG-FIT はアンカー（$\mathcal{L}_{anc}$）や階層制約（$\mathcal{L}_{hier}$）で「意味的に近いものは近い」方向へ引っ張るため、
  特定の target relation に関しては「スコアが上がりやすい」ことがあり得る。

2) **ランキング指標（Hits/MRR）が悪化した点の解釈（重要）**
- KG-FIT の総目的は $\mathcal{L}=\zeta_1\mathcal{L}_{hier}+\zeta_2\mathcal{L}_{anc}+\zeta_3\mathcal{L}_{link}$ で、
  「リンク予測（ランキング）最適化」以外の目的が強く入る。
- そのため、ハイパーパラメータ（$\zeta$、混合率 $\rho$、近傍クラスタ数、階層の粗さ等）が未調整だと、
  **targetには効くが、全体ランキング（filtered評価）ではトレードオフ**になる可能性がある。
- 今回「target score は改善、MRR/Hits は悪化」という同時発生は、まさにそのトレードオフを疑うべき形。

3) **スコア正規化の注意（sigmoid飽和）**
- raw score のスケールが大きい場合、sigmoid 正規化は飽和しやすく、before/after の差が 0 に張り付く。
- 今回 PairRE 側で sigmoid Δ が 0 になったのはこの典型で、
  「score改善の検出」用途では minmax(train)（各モデルの train score min/max によるスケーリング）の方が有用だった。

4) **target triples の unknown（117中 11 を除外）の意味**
- before/after の TriplesFactory 上で unknown entity/relation になる target が一定数ある。
- これは “比較の公正性” に直結するため、次回以降は、
  - dataset_dir（entities/relations定義）
  - model_before_dir / model_after_dir の training_triples（マッピング）
  が target_triples と厳密整合していることを事前チェックした上で実行するのが望ましい。

### 次のアクション案（target改善を維持しつつランキング悪化を切り分ける）

- **KG-FIT(PairRE) の正則化強度のスイープ**: $\zeta_1,\zeta_2,\zeta_3$（もしくは対応する設定パラメータ）を調整し、target改善と MRR の両立点があるか探索。
- **階層/近傍の感度**: seed階層の粗さ（tau選択）や neighbor_k を変えて、過度な凝集/分離でランキングが崩れていないかを見る。
- **incident triples の影響切り分け**: 既に導線はあるため、KG-FIT(PairRE) で incident OFF（または上限K）を固定し、target/MRR の変化を再測定する。

---

## 結論（観測→含意→次アクション）

### 観測

- 本設定（arm=10/UCB/priors=off/ep=100）では、target triples の再スコア（minmax(train)）は KG-FIT(PairRE) が改善優勢、TransE は悪化優勢だった。
- 一方で ranking 指標（MRR）では、TransE が改善し、KG-FIT(PairRE) は悪化した。

### 含意

- 「target score 最適化」を主目的に置くなら KG-FIT(PairRE) は有望だが、リンク予測ランキング（Hits/MRR）とはトレードオフが生じ得る。
- 比較軸（target score vs ranking）を先に明示しないと、同じ実験結果でも結論がぶれる。

### 次アクション

- KG-FIT(PairRE) の正則化強度・階層/近傍（neighbor_k）をスイープして、target改善と MRR の両立点があるか探索する。
- target triples の unknown 除外（117中11）を事前チェックし、before/afterで比較対象が一致するようにデータとモデルの整合性を担保する。

## 参照

- `run_full_arm_pipeline.py` の仕様: docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md
- arm pipeline 実装仕様（priors/incident 含む）: docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md
- KG-FIT標準仕様（成果物・運用）: docs/rules/RULE-20260119-TAKG_KGFIT-001.md
