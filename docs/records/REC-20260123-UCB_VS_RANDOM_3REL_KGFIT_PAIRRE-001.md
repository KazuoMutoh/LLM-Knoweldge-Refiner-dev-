# REC-20260123-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001: KG-FIT(PairRE) arm=10/UCB の改善がランダム追加（seed=0）より良いかを3 relation（incident除去データ）で比較する実験計画

作成日: 2026-01-23
最終更新日: 2026-02-01

## 運用メモ（このファイルの位置付け）

- 本ドキュメントは **実験計画書（実行手順の定義）** であり、この時点で実験を開始しない。
- 実行は、別途「実行開始の合意（Go）」を取ってから行う。

## 目的

既存の nationality 実験計画/結果（UCB vs Random）を、incident triples 除去（target entity incidents removed）で作成した以下3 relation のテストデータに拡張し、
**UCB（arm-runで選ばれた追加）** が **同数ランダム追加（seed=0）** より target 改善に寄与するかを比較する。

対象 relation:
- `/people/person/nationality`
- `/people/person/place_of_birth`
- `/people/person/profession`

主指標:
- target triples の再スコア比較（minmax(train) 正規化）による Δ（after-before）

副指標:
- summary.json のランキング指標（MRR/Hits@k）の before/after 変化

参照（同様の実験のベース）:
- [docs/records/REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001.md](REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001.md)
- [docs/records/REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001.md](REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001.md)

## 比較条件

各relationについて、以下2条件を比較する。

### 条件U（新規）: UCB（arm=10, priors=off）

- `run_full_arm_pipeline.py` を用いて、rule生成〜arm生成〜arm-run〜after再学習/評価までを通し実行する。
- Selector: UCB

### 条件R（新規）: Random add（seed=0）

- UCB run の結果として「実際に追加されたトリプル数（重複除去後）」を $N$ とし、
  `train_removed.txt` から **同数 $N$** を **without replacement（非復元抽出）** でサンプリングして train に追加、after を再学習/評価する。
- Random の反復は **seed=0 のみ** とする（計算量を抑える）。

## 実験条件（固定）

参照実験（nationality v3でのKG-FIT(PairRE)）と揃える。

- KGE: **KG-FIT(PairRE)**
- embedding config: `/app/config_embeddings_kgfit_pairre_fb15k237.json`
- num_epochs: 100
- arm: 10（singleton armのみ）
  - `--n_rules 10` かつ `--k_pairs 0`
- selector: `--selector_strategy ucb`
- priors: off（厳密化のため `--disable_relation_priors` を付ける）
- incident triples: off（`train_removed.txt` からの「incident candidate triples 自動追加」を無効化するため `--disable_incident_triples` を付ける）
- after_mode: retrain

## 使用データセット（incident除去）

今回の対象は、target entity 自身の incident triples を train/valid/test から除去したテストデータ（*_head_incident_v1）である。
KG-FITが必要とするテキスト資源と `.cache/kgfit` を含む `_kgfit` ディレクトリを用いる。

### Dataset一覧

- nationality:
  - dataset_dir: `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit`
  - target_triples: `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt`
  - candidate_triples: `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit/train_removed.txt`

- place_of_birth:
  - dataset_dir: `/app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit`
  - target_triples: `/app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/target_triples.txt`
  - candidate_triples: `/app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit/train_removed.txt`

- profession:
  - dataset_dir: `/app/experiments/test_data_for_profession_head_incident_v1_kgfit`
  - target_triples: `/app/experiments/test_data_for_profession_head_incident_v1_kgfit/target_triples.txt`
  - candidate_triples: `/app/experiments/test_data_for_profession_head_incident_v1_kgfit/train_removed.txt`

## 実行手順（relationごとに実施）

以下の手順を、3 relation それぞれで行う。

### Step 0) before model の学習（KG-FIT(PairRE), ep=100）

UCB/Random の比較で before を固定するため、各 dataset_dir ごとに before model を学習して保存する。

```bash
# 例: nationality
PAIRRE_BEFORE_DIR=/app/models/20260123/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100

python3 /app/scripts/train_initial_kge.py \
  --dir_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
  --output_dir "$PAIRRE_BEFORE_DIR" \
  --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
  --num_epochs 100 \
  --skip_evaluation
```

### Step 1) 条件U（UCB）の実行

`run_full_arm_pipeline.py` を用いてUCBを実行する。

```bash
# 例: nationality
RUN_DIR=/app/experiments/20260123/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_20260123a

python3 /app/run_full_arm_pipeline.py \
  --run_dir "$RUN_DIR" \
  --model_dir /app/models/20260123/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100 \
  --target_relation /people/person/nationality \
  --dataset_dir /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/train_removed.txt \
  --n_rules 10 \
  --k_pairs 0 \
  --n_iter 25 \
  --k_sel 3 \
  --n_targets_per_arm 20 \
  --selector_strategy ucb \
  --disable_relation_priors \
  --disable_incident_triples \
  --after_mode retrain \
  --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
  --num_epochs 100 \
  2>&1 | tee "$RUN_DIR/run.log"
```

UCB実行後に確認する成果物:
- `$RUN_DIR/arm_run/retrain_eval/summary.json`
- `$RUN_DIR/arm_run/retrain_eval/updated_triples/added_triples.tsv`

この run の `summary.json` に出力される「実際に追加に使われた集合サイズ」（例: `n_added_used`）を $N$ とし、次の Random 条件で「同数追加」を行う。

### Step 2) 条件R（Random seed=0）の実行

#### 2-1) サンプル作成（without replacement）

`train_removed.txt` から $N$ 本を seed=0 で非復元抽出し、arm-run互換の位置に配置する。

補足:
- Random 条件では arm pipeline を走らせないため、UCB側に存在する「incident candidate triples の自動追加」は発生しない（= サンプルしたものだけを追加する）。

```bash
# 例: nationality
RUN_DIR=/app/experiments/20260123/exp_random_addN_priors_off_kgfit_pairre_ep100_nationality_head_incident_v1_seed0_20260123a
ARM_RUN_DIR="$RUN_DIR/arm_run"

# N は Step1 の added_triples.tsv から決める
#  - `scripts/sample_random_triples.py` は入力の重複行を除去した上で rng.sample する
python3 /app/scripts/sample_random_triples.py \
  --input_tsv /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/train_removed.txt \
  --output_tsv "$ARM_RUN_DIR/iter_1/accepted_added_triples.tsv" \
  --n <N> \
  --seed 0
```

#### 2-2) 再学習・評価

```bash
python3 /app/retrain_and_evaluate_after_arm_run.py \
  --run_dir "$ARM_RUN_DIR" \
  --dataset_dir /app/experiments/test_data_for_nationality_head_incident_v1_kgfit \
  --target_triples /app/experiments/test_data_for_nationality_head_incident_v1_kgfit/target_triples.txt \
  --model_before_dir /app/models/20260123/fb15k237_kgfit_pairre_nationality_head_incident_v1_before_ep100 \
  --after_mode retrain \
  --embedding_config /app/config_embeddings_kgfit_pairre_fb15k237.json \
  --num_epochs 100 \
  2>&1 | tee "$RUN_DIR/run.log"
```

出力:
- `$RUN_DIR/arm_run/retrain_eval/summary.json`
- `$RUN_DIR/arm_run/retrain_eval/model_after/`
- `$RUN_DIR/arm_run/retrain_eval/updated_triples/`

## 評価・比較方法

### 1) summary.json 指標

UCB vs Random(seed=0) の差分を比較する。
- `target_score_change`（raw; ただし主張は次項の minmax(train) を主とする）
- `mrr_change`, `hits_at_k_change`
- `n_triples_added` / `n_added_used` 等（runの出力に従う）

### 2) target triples 再スコア（主指標）

「minmax(train) による target score Δ」の比較を行う。

- 参照スクリプト（暫定）: `/app/scripts/compute_minmax_rescore_section.py`
- 方針:
  - 各relationについて、before model を固定し、after model（UCB/Random seed=0）で target triples を再スコア
  - 各モデルの train score の min-max で 0..1 に正規化し、Δ（after-before）を算出
  - before/after の双方で unknown な target を除外し、比較は共通集合（intersection）で行う

成果物（想定）:
- `/app/experiments/20260123/compare_ucb_vs_random_seed0_<relation>_head_incident_v1_20260123a.md`
  - summary.json 比較 + minmax(train) セクション

## リスク・注意点

- target coverage（unknown除外）:
  - before/afterでknownなtarget数が一致しない可能性がある。
  - minmax(train) の比較は共通集合（intersection）で行い、usable targets数を必ず併記する。

- Randomはseed=0のみ:
  - ランダムの分散推定ができない点は明確に限界として記録する。

- 実行時間:
  - before学習（ep=100）×3、UCB（arm-run + after retrain）×3、Random after retrain ×3。
  - まずは1 relation から実行し、見積りを更新してから並列/順次を決める。

## 成果物（保存場所）

- 実行ディレクトリ: `/app/experiments/20260123/`
- relationごとの run_dir（命名例）:
  - UCB:
    - `exp_ucb_arm10_priors_off_kgfit_pairre_ep100_<relation>_head_incident_v1_20260123a`
  - Random(seed=0):
    - `exp_random_addN_priors_off_kgfit_pairre_ep100_<relation>_head_incident_v1_seed0_20260123a`

## 次のTODO

- [ ] 本計画のGo合意
- [ ] 3 relationの before model 学習（ep=100）
- [ ] 3 relationの UCB 実行（arm=10, priors=off）
- [ ] 各UCBの $N$ を決定し、Random(seed=0) を実行
- [ ] minmax(train) 正規化による target 再スコア比較レポートを作成

---

## 結論（観測→含意→次アクション）

### 観測

- incident除去データ（*_head_incident_v1）を対象に、UCB（arm-run）と同数Random（seed=0）を3 relationで比較する実験計画を定義した。
- 主指標は minmax(train) 正規化による target 再スコアΔ、副指標は Hits/MRR の before/after 変化とした。

### 含意

- 同一プロトコルを複数relationに適用することで、特定relation依存の偶然ではなく「選別が効く条件」を見分けやすくなる。
- Randomをseed=0のみとするため、結論は探索的（分散推定が弱い）になり得る。

### 次アクション

- まず1 relation（nationality）でbefore学習→UCB→Randomを通し実行し、計算時間/安定性を見積もった上で3 relationへ展開する。
- 可能ならRandomを複数seedへ拡張し、分散を見積もる。
