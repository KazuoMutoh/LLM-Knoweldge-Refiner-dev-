# REC-20260121-UCB_KGFIT_INTERVENTIONS-001: UCB（full pipeline, KG-FIT）に対する介入（nationality除外 / tail上限）検証

作成日: 2026-01-21  
最終更新日: 2026-02-01

参照:
- [REC-20260118-FULL_PIPELINE_RESULTS-001](REC-20260118-FULL_PIPELINE_RESULTS-001.md): 元の full pipeline 結果（TransE）
- [REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001](REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001.md): 同一の追加前/追加後KGを用いた KG-FIT 再評価（3条件）
- [REC-20260119-TAKG_KGFIT-001](REC-20260119-TAKG_KGFIT-001.md): KG-FIT採用の背景/目的（TAKG移行）
- [RULE-20260119-TAKG_KGFIT-001](../rules/RULE-20260119-TAKG_KGFIT-001.md): KG-FITバックエンド運用標準

成果物（本検証の出力）:
- 比較サマリ: [compare.json](../../experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_ep100_ucb_interventions_20260121a/compare.json)
- 実行ログ: [run_resume.log](../../experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_ep100_ucb_interventions_20260121a/run_resume.log)
- 実行スクリプト: [rerun_ucb_interventions_kgfit.py](../../tmp/debug/rerun_ucb_interventions_kgfit.py)

成果物（PairRE+KG-FIT, ep=2 再実行の出力）:
- 比較サマリ: [compare.json](../../experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_pairre_ep2_ucb_interventions_20260121b/compare.json)
- 実行スクリプト: [rerun_ucb_interventions_kgfit_pairre.py](../../tmp/debug/rerun_ucb_interventions_kgfit_pairre.py)

成果物（PairRE+KG-FIT, ep=100 再実行の出力）:
- 比較サマリ: [compare.json](../../experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_pairre_ep100_ucb_interventions_20260121c_ep100/compare.json)
- 実行スクリプト: [rerun_ucb_interventions_kgfit_pairre.py](../../tmp/debug/rerun_ucb_interventions_kgfit_pairre.py)

---

## 0. 目的

「TransEベース KG-FIT の限界か？」を切り分けるため、まずモデル変更は行わず **同一の TransE+KG-FIT** のまま、追加トリプルの性質（特に nationality の集中や tail ハブ化）を抑制する介入を入れて再評価する。

本記録では、[REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001](REC-20260120-FULL_PIPELINE_RESULTS_KGFIT-001.md) の **UCB条件**（arm10+priors）の追加後KG（updated_triples）を対象に、次の2介入の効果を検証する。

---

## 1. 対象と前提

対象（UCB条件）:
- run_dir: `/app/experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run`
- dataset_dir（追加前KG）: `/app/experiments/test_data_for_nationality_v3`
- updated_triples_dir（追加後KG）: `<run_dir>/retrain_eval/updated_triples`
- target relation: `/people/person/nationality`
- epochs: 100
- KG-FIT設定: `/app/config_embeddings_kgfit_fb15k237.json`
- before model（共有）: `/app/models/20260120/fb15k237_kgfit_transe_nationality_before_ep100_fullpipeline_001`

評価:
- `simple_active_refine.evaluation.IterationEvaluator`
- target score: `KnowledgeGraphEmbedding.score_triples()` 平均（raw）
- Hits@1/3/10, MRR: `KnowledgeGraphEmbedding.evaluate()`（test split）

注意:
- target_triples は 117件だが、KG-FIT scoring では unknown entity/relation により 11件が除外される（scorable subset=106）。

---

## 2. 介入の定義

共通方針:
- 既存の `updated_triples/train.txt` そのものを直接編集せず、
  - `evidence_added_only := set(updated_train) - set(baseline_train)`
  を推定し、**追加分のみ**をフィルタした上で `baseline_train ∪ filtered_evidence` で新しい updated_triples を再構成する。

### 2.1 介入A: nationality 追加の除外

- 追加分（evidence_added_only）から、predicate が `/people/person/nationality` のトリプルを除外する。
- 目的: 「* , nationality, 特定の国」が大量増殖している場合の影響を除去する。

### 2.2 介入B: tail上限（tail-cap）

- 追加分（evidence_added_only）を tail entity ごとに上限Kでクリップする。
- 本検証では K=20 を採用。
- 目的: 特定 tail（国）への接続が過度に集中するハブ化を抑制する。

---

## 3. 結果

比較対象（元のUCB, KG-FIT再評価）:
- 出力: `/app/experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_ep100_full_pipeline_results_kgfit_001/summary.json`

介入結果出力:
- `/app/experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_ep100_ucb_interventions_20260121a/`

### 3.1 指標（before→after, Δ）

| 条件 | n_added | target score Δ | Hits@1 Δ | Hits@3 Δ | Hits@10 Δ | MRR Δ |
|---|---:|---:|---:|---:|---:|---:|
| 元UCB（KG-FIT） | 10185 | -3.3548 | +0.0294 | -0.0172 | -0.0637 | +0.0144 |
| 介入A（nationality除外） | 10036 | -9.4501 | +0.0221 | -0.0343 | -0.0172 | +0.0121 |
| 介入B（tail-cap=20） | 8069 | -8.2304 | -0.0147 | -0.0392 | -0.0490 | -0.0212 |

補足（介入強度）:
- 介入Aは n_added が 10185→10036（差分149）で、UCB条件では nationality 追加がそもそも多くないことを示唆する。
- 介入Bは n_added が 10185→8069 まで減る（-2116）ため、追加トリプルの多くが特定 tail に偏っていた可能性がある。

---

## 4. 考察（切り分け）

### 4.1 「モデル限界」より先に疑うべき点

- UCB条件では、単純な “nationality 追加の全除外” では改善は見られなかった。
  - 対象の nationality 追加が少ない（149件程度）ため、主要因に当たりにくい。
- 一方で tail-cap（K=20）は明確に悪化（MRR/Hits@1が低下）した。
  - これは「UCBで追加されたトリプルには有益な成分も多く、無差別に削ると性能が落ちる」ことを示唆する。

このため、本条件（UCB）に限って言えば、現時点の証拠では
- 「TransEベース KG-FIT の限界」が主因
というより
- 「追加トリプルは比較的多様で、雑なハブ抑制が過剰」
の可能性が高い。

### 4.2 次の一手（より“切り分け”に効く介入）

今回の介入Bは“全relation一律”でtailを抑えたため、必要な構造情報も一緒に削ってしまうリスクが高い。

切り分け目的（ハブ化のみを狙う）に合わせて、次は以下が優先:
- **relation限定 tail-cap**: `/people/person/nationality` の追加だけを tail-cap（例: K=20）
- **tail限定 cap**: 追加が集中している tail（例: `/m/09c7w0`）に対してのみ上限を入れる
- **LLM条件側で同じ介入**: ハブ化が強く観測された条件（LLM-policy）に適用し、改善するかで “データ偏り vs モデル” をより明確に分離する

### 4.3 補足: “最悪化した target triple” の具体例（A/B/C どれで落ちたか）

本節は「target triple のスコアが落ちる機序」を具体例で確認するための補足であり、UCB介入そのものの結果ではない。

対象データ（LLM rerun, KG-FIT 再評価の per-target 分析出力）:
- per-target Δ: [per_target_scores.csv](../../tmp/debug/kgfit_target_delta/llm_rerun/per_target_scores.csv)
- 追加トリプル（追加分）: [added_triples.tsv](../../experiments/20260118/exp_B4_llm_25_rerun_20260118b/arm_run/retrain_eval/updated_triples/added_triples.tsv)

ここで A/B/C は以下の意味で整理する:
- **A（同一headでの “競合tail” 追加）**: 例: 同一人物headに対し、別の国tailの nationality が追加される（同一headの候補tailが増えて、ランキングが割れる）
- **B（tailハブ化 / tail側集中）**: 例: 特定の国tailに nationality（や country/contains 等）が大量に集まり、tail embedding と relation embedding が「ハブに合わせた妥協点」に寄る
- **C（head側の incident 追加で引っ張られる）**: 例: headに対して大量の追加が入り、head embedding が別制約（他relation）に強く引っ張られる

#### 例1: tailハブ化（Bが支配的）

target: `(/m/0dr5y, /people/person/nationality, /m/09c7w0)`
- Δscore = -118.13（before=-2555.95 → after=-2674.08）
- incident 追加数（per-target 出力）: head=1 / tail=304
- 追加トリプル側の観測（added_triples.tsv）:
  - tail=`/m/09c7w0` を object に取る追加が 276 本
  - うち `/people/person/nationality` が 142 本、`/film/film/country` が 56 本など

判定:
- **Bが主因**（tail=`/m/09c7w0` が強いハブとして学習を支配し、同tailを取る target triple の score が大きく下がる）
- Aは観測されない（この head を subject とする追加 nationality は 0）
- Cは弱い（head側 incident は 1）

#### 例2: tailハブ化（B）+ head側追加（Cは軽微）

target: `(/m/04__f, /people/person/nationality, /m/09c7w0)`
- Δscore = -98.34（before=-2523.82 → after=-2622.16）
- incident 追加数（per-target 出力）: head=15 / tail=304
- head側には `/award/...` や `/film/actor/...` など複数relationの追加が入り（head_top3参照）、tail=`/m/09c7w0` 側も上記の通りハブ

判定:
- **Bが主因** + **Cが上乗せ**（tailハブ化で relation/tail が引っ張られ、さらに head embedding も別relationの追加で微調整が入る）
- Aは観測されない（この head を subject とする追加 nationality は 0）

#### 例3: “局所には何も足していないのに落ちる” ケース（Bの波及）

target: `(/m/0465_, /people/person/nationality, /m/014tss)`
- Δscore = -75.06（before=-2482.65 → after=-2557.71）
- incident 追加数（per-target 出力）: head=1 / tail=1
- 追加トリプル側の観測（added_triples.tsv）:
  - tail=`/m/014tss` を object に取る追加は 0 本（=この tail 自体はハブ化していない）

判定:
- **Bの“波及”**が主因（`/m/09c7w0` のような大ハブと大量の nationality 追加が、relation embedding や負例サンプリングの分布を押し、直接関係しない triple の score も下げ得る）
- A/C はこの例では強くない（head/tail とも局所の追加がほぼない）

#### 例4: 中規模のtail集中（Bが中程度）

target: `(/m/05bp8g, /people/person/nationality, /m/03_3d)`
- Δscore = -45.46（before=-2495.40 → after=-2540.86）
- incident 追加数（per-target 出力）: head=3 / tail=10

判定:
- **Bが中程度**（tail側に一定数の incident 追加があり、tail embedding が押される）
- Cも軽微（head側 incident は 3）
- Aは観測されない（この head を subject とする追加 nationality は 0）

観測範囲（LLM rerun の worst-case 上位）では、**A（同一headの競合tail追加）よりも、B（tailハブ化/その波及）が支配的**だった。

---

## 更新履歴

- 2026-01-21: 新規作成。UCB条件の updated_triples に対し、nationality除外・tail-cap の2介入で TransE+KG-FIT を再評価した結果を記録。
- 2026-01-21: 追記。PairRE+KG-FIT（ep=2）で同一介入（A/B）を再実行し、compare.json として記録。
- 2026-01-22: 追記。PairRE+KG-FIT（ep=100）で同一介入（A/B）を再実行し、compare.json として記録。

---

## 5. PairRE+KG-FIT での再実行（ep=2, 2026-01-21）

本RECの当初目的は「モデル変更は行わず TransE+KG-FIT で介入効果を切り分ける」だったが、KGE（Interaction）差分の影響も併せて見るため、**同一データ/同一介入**を PairRE+KG-FIT でも再実行した。

### 5.1 対象と前提

- run_dir: `/app/experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run`
- dataset_dir（追加前KG）: `/app/experiments/test_data_for_nationality_v3`
- original updated_triples_dir（追加後KG）: `<run_dir>/retrain_eval/updated_triples`
- target relation: `/people/person/nationality`
- epochs: 2（スモーク / 収束前のため注意）
- embedding_config: `/app/config_embeddings_kgfit_pairre_fb15k237.json`

出力:
- out_dir: `/app/experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_pairre_ep2_ucb_interventions_20260121b/`
  - model_before: `<out_dir>/model_before/`（PairRE+KG-FIT を新規に学習）
  - 介入A: `<out_dir>/exclude_people_person_nationality/`
  - 介入B: `<out_dir>/tailcap_20/`

実行:
- スクリプト: [rerun_ucb_interventions_kgfit_pairre.py](../../tmp/debug/rerun_ucb_interventions_kgfit_pairre.py)
- コマンド例:
  - `PYTHONPATH=/app /app/.venv/bin/python /app/tmp/debug/rerun_ucb_interventions_kgfit_pairre.py`

### 5.2 結果（PairRE+KG-FIT, before→after, Δ）

PairREの before（共通）:
- Hits@1=0.1250, Hits@3=0.2868, Hits@10=0.4485, MRR=0.2327
- target score(before)=-130.0855

| 条件 | n_added | target score Δ | Hits@1 Δ | Hits@3 Δ | Hits@10 Δ | MRR Δ |
|---|---:|---:|---:|---:|---:|---:|
| 介入A（nationality除外） | 10036 | -0.0640 | +0.0025 | -0.0172 | +0.0245 | -0.0008 |
| 介入B（tail-cap=20） | 8069 | -6.9278 | +0.0245 | -0.0025 | +0.0343 | +0.0244 |

補足:
- 介入Aは、PairREでも効果が小さい（n_added の差分も小さい）。
- 介入Bは、ep=2 の範囲では Hits/MRR を押し上げる一方で、target score（raw）は大きく低下した。

### 5.3 注意点（比較の限界）

- 本節は epochs=2 のスモークであり、ep=100 の TransE+KG-FIT（第3章）と **定量比較はできない**（収束度/ばらつきが異なる）。
- “PairREに変えて同実験を再度”を本比較として扱うには、同一epochs（例: 100）での再実行が必要。

### 5.4 結果（PairRE+KG-FIT, ep=100, 2026-01-22）

出力:
- out_dir: `/app/experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_pairre_ep100_ucb_interventions_20260121c_ep100/`
- 比較サマリ: [compare.json](../../experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval_kgfit_pairre_ep100_ucb_interventions_20260121c_ep100/compare.json)

PairRE の before（共通）:
- Hits@1=0.1348, Hits@3=0.2917, Hits@10=0.6201, MRR=0.2717
- target score(before)=-488.3391

| 条件 | n_added | target score Δ | Hits@1 Δ | Hits@3 Δ | Hits@10 Δ | MRR Δ |
|---|---:|---:|---:|---:|---:|---:|
| 介入A（nationality除外） | 10036 | +9.9581 | +0.0025 | +0.0270 | -0.0221 | +0.0069 |
| 介入B（tail-cap=20） | 8069 | -3.3221 | -0.0270 | +0.0221 | +0.0196 | -0.0078 |

所見（PairRE+KG-FIT, ep=100 の範囲）:
- 介入Aは、Hits@3 と MRR が改善する一方で Hits@10 は低下した。
- 介入Bは、Hits@3/Hits@10 は改善したが Hits@1/MRR が悪化した。

---

## 結論（観測→含意→次アクション）

### 観測

- UCB条件（TransE+KG-FIT, ep=100）では、nationality除外（介入A）は追加件数差が小さく、指標改善にも直結しなかった。
- tail-cap（介入B, K=20）は追加件数を大きく減らす一方で、MRR/Hits@1の悪化など「削り過ぎ」による劣化が観測された。
- PairRE+KG-FITでも、介入A/Bは指標にトレードオフ（Hits@k間・MRR・target score）を生み、単純な一律抑制では安定しない。

### 含意

- 「モデル限界」より先に、追加トリプルの偏り（特にtail側集中）をどう抑えるかが効くが、全relation一律のtail-capは過剰介入になりやすい。
- 有害成分（ハブ化/ノイズ）だけを狙い撃ちする介入（relation限定・tail限定）が必要。

### 次アクション

- relation限定 tail-cap（例: `/people/person/nationality` の追加に限定してKを入れる）を優先して再評価する。
- 追加集中の強いtailのみを対象にしたtail限定capを試し、「B（tailハブ化）」の寄与を切り分ける。
- 介入はUCBだけでなく、ハブ化が強く観測された条件（LLM-policy等）にも適用して、データ偏り vs モデル要因を分離する。
