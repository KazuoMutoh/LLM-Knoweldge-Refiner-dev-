# REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001: LLM-policy run（nationality）の iter=3 / 最終iter の追加トリプルで KGE を再学習し、target score と Hits を評価する実験計画

- 作成日: 2026-01-29
- 最終更新日: 2026-02-01
- 参照:
  - LLM-policy run（再学習なし・取得結果）: [REC-20260129-LLM_POLICY_NATIONALITY_TRIPLE_ACQUISITION-001](REC-20260129-LLM_POLICY_NATIONALITY_TRIPLE_ACQUISITION-001.md)
  - 再学習+評価スクリプト: [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)

## 目的

LLM-policy arm-run（nationality/head_incident_v1）の出力から、
- iter=3 の「そのiterationで取得された追加トリプル」
- 最終iter（例: iter=25）の「そのiterationで取得された追加トリプル」

をそれぞれ KG に加えて KGE を再学習し、
- target triples の平均スコア変化
- KGE の評価指標（Hits@k, MRR）の変化

を比較する。

さらに、同数トリプルをランダムに選んで加える対照実験を入れ、
「iter=3（または最終iter）の取得トリプルが、単なるランダム追加より有効か」を検証する。

## 実験条件（固定）

### 対象データセット

- dataset_dir:
  - `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit`
- target_triples:
  - `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt`
- ランダムサンプリング元（候補集合）:
  - `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit/train_removed.txt`

### before model（固定）

- model_before_dir:
  - `/app/models/20260125_rerun1/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100`

### KGE再学習（after）の設定

- embedding_config（KG-FIT + PairRE）:
  - `/app/config_embeddings_kgfit_pairre_fb15k237.json`
- num_epochs:
  - 100（before と同等の学習回数に揃える）

補足:
- KG-FIT の場合、dataset_dir 配下に `.cache/kgfit/` が必要（存在しない場合は事前計算を実施）。

### 評価方法

- `retrain_and_evaluate_after_arm_run.py` を用い、以下を出力する:
  - `retrain_eval/evaluation/iteration_metrics.json`
  - `retrain_eval/summary.json`
- 主要指標:
  - `target_score_change`
  - `hits_at_1_change`, `hits_at_3_change`, `hits_at_10_change`, `mrr_change`

### 除外（リーク防止）

- 原則として target predicate は追加データから除外する:
  - `--exclude_predicate /people/person/nationality`

（今回の LLM-policy run では nationality 自体の追加が 0 件だが、対照実験も含めてルール化しておく。）

## 実験変数（4条件）

出力ディレクトリは以下のように作る（例）:

- `/app/experiments/20260129_iter3_final_kge_eval_nationality/`
  - `cond1_iter3_actual/`
  - `cond2_iter3_random/`
  - `cond3_final_actual/`
  - `cond4_final_random/`

比較したい条件:

1. **cond1: iter=3 の取得トリプルを追加**
2. **cond2: cond1 と同数のトリプルをランダムに選択して追加**
3. **cond3: 最終iter（例: iter=25）の取得トリプルを追加**
4. **cond4: cond3 と同数のトリプルをランダムに選択して追加**

## 入力（トリプル集合の作り方）

### 取得トリプル（actual）の定義

「そのiterationで取得された追加トリプル」を使う。

- iter=3: 
  - `/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run_retry_bg_20260129_202535/iter_3/accepted_added_triples.tsv`
- 最終iter（例: iter=25）:
  - `/app/experiments/20260129_llm_policy_nationality_head_incident_v1_from_rerun1/arm_run_retry_bg_20260129_202535/iter_25/accepted_added_triples.tsv`

補足:
- `accepted_added_triples.tsv` は evidence + incident の和集合。
- 各iterationの「増分」を比較したいので、**累積（iter1..kの合算）ではなく、そのiter単体ファイル**を使う。

### ランダム（random）の定義

サンプリング元: `train_removed.txt`

フィルタ規則（提案）:
- 重複除去（同一(s,r,o)は1回）
- 既存 train.txt と同一のトリプルは除外（追加しても変化が出にくいため）
- `exclude_predicate`（nationality）は除外

サンプリング数:
- cond2: `|iter_3/accepted_added_triples.tsv|` と同数
- cond4: `|iter_25/accepted_added_triples.tsv|` と同数

再現性:
- `random_seed=0` を固定し、選ばれたトリプル一覧（TSV）と seed を `cond*/iter_1/` 配下に保存する。

## 実行手順

### Step 0: 事前に件数を確認

- iter3 と最終iterの追加件数を確認して、random側のサンプル数を確定する。

### Step 1: 4つの「擬似 run_dir」を作る

`retrain_and_evaluate_after_arm_run.py` は `run_dir` 配下の `iter_*/accepted_added_triples.tsv` を集約する。
今回の比較は「単一iterationの増分」なので、各条件について以下の形の最小 run_dir を作る:

- `condX/iter_1/accepted_added_triples.tsv`

（※ `iter_1` という名前でよく、スクリプトは集約時に iter番号を見ているだけなので、単一iterationとして扱える。）

### Step 2: 各条件で KGE を再学習し、評価する

condごとに以下を実行（例: cond1）:

- `python retrain_and_evaluate_after_arm_run.py \
  --run_dir /app/experiments/20260129_iter3_final_kge_eval_nationality/cond1_iter3_actual \
  --dataset_dir /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
  --model_before_dir /app/models/20260125_rerun1/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100 \
  --after_mode retrain \
  --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
  --num_epochs 100 \
  --exclude_predicate /people/person/nationality \
  --force_retrain`

同様に cond2..cond4 を回す。

### Step 3: 集計

各condの `retrain_eval/summary.json` を集めて、以下の比較表を作る:

- iter3_actual vs iter3_random
- final_actual vs final_random
- iter3_actual vs final_actual

（必要なら scripts/ に集計スクリプトを作成して Markdown で残す。）

## 期待される成果物

- `cond*/iter_1/accepted_added_triples.tsv`
- `cond*/retrain_eval/summary.json`
- `cond*/retrain_eval/evaluation/iteration_metrics.json`

## 成功条件（受け入れ基準）

- 4条件すべてで KGE の再学習が完走し、`summary.json` が生成される。
- 比較表が作れ、少なくとも以下を確認できる:
  - `target_score_change` の符号/大きさ
  - `Hits@k` / `MRR` の変化
  - actual が random を上回るか（上回らないなら、その理由仮説を立てられる）

## リスク / 注意点

- 学習コスト: KG-FIT + ep=100 は時間がかかる。まずは ep=10 の pilot を行い、問題なければ ep=100 に上げる運用も可。
- ランダムの分散: 可能なら seed を複数（例: 0,1,2）にして、random対照のばらつきを見積もる。
- 追加トリプルがノイズ過多の場合、Hits/MRR が悪化する可能性がある（その場合でも「悪化する」こと自体が重要な観測）。

---

## 実験結果（追記: 累積union版 / ep=100 本番、2026-01-31 完走）

実験出力:

- `/app/experiments/20260130_iter3_final_kge_eval_nationality_cumulative/`

主要な結果（before共通）:

- target score (before): -736.4064
- Hits@10 (before): 0.290229
- MRR (before): 0.128665

比較（actual - random）:

- iter1..3 union（159件）: target score Δ=+19.8420, Hits@10 Δ=+0.013713, MRR Δ=+0.004195
- iter1..final union（887件）: target score Δ=-14.9608, Hits@10 Δ=+0.000147, MRR Δ=-0.002824

## 結論（観測→含意→次アクション）

### 観測

- 単発iteration版（iter3=73件/final=9件）では定義が「累積（iter1..k）」と不整合だったため、累積union版で再評価した。
- 累積union版（ep=100）では、iter1..3（159件）の actual は同数randomより target score/Hits@10/MRR の劣化を抑えつつ改善幅が大きかった。
- 一方で iter1..final（887件）の actual は、同数randomに対して target score と MRR が劣後した。

### 含意

- 「少量〜中量（iter1..3）での有益な証拠追加」は nationality の target score 改善に寄与しうる。
- しかし「大量（iter1..final）の無選別追加」は、同数randomよりも劣る（=ノイズ混入/意味不整合の影響が勝つ）ケースがあり得る。

### 次アクション

- final union（887件）の追加トリプルを predicate サブセット（地理系/非地理系など）に分け、どの成分が劣化を作っているか切り分ける。
- random対照は seed を複数に増やして分散を見積もる（特にfinalは件数が大きく、差分が小さいため）。
- 今後の再学習評価は [RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001](../rules/RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001.md) に沿って、定義（incremental/cumulative）を明示した上で実行する。
