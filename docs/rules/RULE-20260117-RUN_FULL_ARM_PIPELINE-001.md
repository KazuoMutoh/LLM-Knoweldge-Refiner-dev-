# RULE-20260117-RUN_FULL_ARM_PIPELINE-001: run_full_arm_pipeline.py 実装仕様（統合ARMパイプライン）

作成日: 2026-01-17
最終更新日: 2026-01-24
更新日: 2026-01-17（early stopper + valid無し運用注意を追記）
更新日: 2026-01-17（arm選択の説明可能性向上のため、dataset_dir/relation2text.txt の任意利用を追記）
更新日: 2026-01-18（Step0: relation priors（X_r）事前計算と arm-run への投入を追記）
更新日: 2026-01-22（Step3: incident triples 追加のON/OFF・上限をCLIから制御可能にする）
更新日: 2026-01-24（Step2: pair-arm の support/Jaccard 計算母集団を candidate/train で切替可能にし、既定値は candidate を維持）

---

## 0. 目的とスコープ

`run_full_arm_pipeline.py` は、arm-based KG refinement を **1コマンドで end-to-end 実行**する統合ランナーである。
本スクリプトは、次の4工程を **同一 run ディレクトリ配下**に集約し、成果物パスを次工程へ安全に受け渡す。

（任意）Step 0. relation priors（$X_r$）計算（witness のKGE-friendly重み付け用）

1. 初期ルールプール生成（AMIE+規則抽出 → 上位Nルール選択）
2. 初期armプール生成（ルール組 = arm を作る）
3. arm駆動の反復精錬（arm選択 → evidence取得 → witness/衝突で代理報酬 → KG更新）
4. arm-run結果の集約 → updated dataset 作成 →（after再学習 or load）→ before/after 評価

本仕様は **現行実装（2026-01-17 時点の `run_full_arm_pipeline.py`）の再実装が可能**な粒度で、I/O、ディレクトリ規約、スキップ条件、主要パラメータを定義する。

非スコープ（このドキュメントでは詳細を定義しない）:
- AMIE+自体のアルゴリズム・品質指標の意味
- witness/衝突評価の数式や妥当性（設計ドキュメント側を参照）
- KGE（PyKEEN）の学習・評価の内部詳細

---

## 1. 位置づけ（アーキテクチャ上の役割）

- `run_full_arm_pipeline.py` は **オーケストレーション専用**であり、各工程のコア処理は既存モジュール/スクリプトへ委譲する。
- 実験の再現性を高めるため、成果物の配置を **runディレクトリ配下の固定構造**に統一する。

関連する設計（背景）:
- arm-based 精錬の考え方（witness/衝突、arm選択戦略、出力規約）: [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- 統合ランナーの設計・実装計画（本スクリプトの元仕様）: [docs/records/REC-20260114-ARM_PIPELINE-001.md](../records/REC-20260114-ARM_PIPELINE-001.md)

---

## 2. 入力（必須）

### 2.1 CLI必須引数

`run_full_arm_pipeline.py` の CLI は、最低限次を要求する。

- `--run_dir`: 実験ルート（成果物を作るトップディレクトリ）
- `--model_dir`: 学習済みKGEディレクトリ（初期ルールプール抽出に使用）
- `--target_relation`: ターゲット関係（例: `/people/person/nationality`）
- `--dataset_dir`: 元データセットディレクトリ（最低 `train.txt` を含む）
- `--target_triples`: ターゲットトリプルTSV（arm生成と取得対象）
- `--candidate_triples`: 候補トリプルTSV（主に `train_removed.txt` を想定）

### 2.2 ファイル形式（TSV）

- `target_triples` / `candidate_triples` / `dataset_dir/train.txt` は、各行が次のTSVである。
  - `head<TAB>relation<TAB>tail`

注意:
- 実装は厳密なバリデーションを行わず、3列でない行はスキップされる（工程により挙動が異なる可能性がある）。

### 2.3 `--embedding_config`（KGE再学習設定）

`Step4: retrain + evaluate` の `after_mode=retrain` では、`--embedding_config` で学習設定JSONを渡す。
この JSON は [scripts/train_initial_kge.py](../../scripts/train_initial_kge.py) と同様に解釈され、
内部で `simple_active_refine.embedding.KnowledgeGraphEmbedding.train_model(...)` に渡される。

期待キー（標準）:
- `model`: 例 `"TransE"`
- `embedding_backend`: `"pykeen" | "kgfit"`（省略時 `pykeen`）
- `kgfit`: `embedding_backend="kgfit"` の場合の追加設定（省略可）

KG-FITを使う場合の前提（重要）:
- `dir_triples`（= dataset_dir）が必須（built-in dataset名は不可）
- `.cache/kgfit/` に事前計算成果物が揃っていること
  - `entity_name_embeddings.npy` / `entity_desc_embeddings.npy` / `entity_embedding_meta.json`
  - `hierarchy_seed.json` / `cluster_embeddings.npy` / `neighbor_clusters.json`

詳細仕様:
- KG-FITバックエンドの標準: [docs/rules/RULE-20260119-TAKG_KGFIT-001.md](RULE-20260119-TAKG_KGFIT-001.md)
- 正則化/近傍K運用: [docs/rules/RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001.md](RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001.md)

---

## 3. 出力（run_dir 配下の固定構造）

`run_dir` を基点に、次のサブディレクトリを固定で使用する。

```
<run_dir>/
  relation_priors/
    relation_priors.json
  rule_pool/
    initial_rule_pool.csv
    initial_rule_pool.pkl
    initial_rule_pool.txt
    amie_tmp/...
  arms/
    initial_arms.json
    initial_arms.pkl
    initial_arms.txt
  arm_run/
    iter_1/
      selected_arms.json
      accepted_evidence_triples.tsv
      pending_hypothesis_triples.tsv
      arm_history.pkl
      arm_history.json
      diagnostics.json
    iter_2/...
    retrain_eval/
      updated_triples/
        train.txt
        valid.txt
        test.txt
        added_evidence.tsv
      model_before/   (デフォルトパス。存在必須)
      model_after/    (after_mode=retrain の場合に作られる)
      evaluation/
      summary.json
```

「キー成果物」の既定パス（ランナー内部でハードコードされる）:
- relation priors（compute指定時）: `<run_dir>/relation_priors/relation_priors.json`
- rule pool: `<run_dir>/rule_pool/initial_rule_pool.pkl`
- arms: `<run_dir>/arms/initial_arms.json`
- retrain/eval summary: `<run_dir>/arm_run/retrain_eval/summary.json`

---

## 4. 工程仕様（run_pipeline 関数）

`run_full_arm_pipeline.py` は import-safe な関数 `run_pipeline(...)` を提供し、CLI `main()` はこれを呼ぶ。

### 4.1 RunnerPaths（内部のパス解決）

- `run_dir` から次を解決する（固定）
  - `rule_pool_dir = run_dir / "rule_pool"`
  - `arms_dir = run_dir / "arms"`
  - `arm_run_dir = run_dir / "arm_run"`
  - `rule_pool_pkl = rule_pool_dir / "initial_rule_pool.pkl"`
  - `arms_json = arms_dir / "initial_arms.json"`
  - `retrain_eval_summary = arm_run_dir / "retrain_eval" / "summary.json"`

### 4.1.5 Step 0: relation priors（X_r）事前計算（任意）

目的:
- witness の寄与を relation-level prior で重み付けし、ハブ関係・KGE非フレンドな関係による水増しを抑制する

実行条件:
- `--compute_relation_priors` が指定されている場合に実行。
- ただし `--relation_priors_path` が明示されている場合は、それを優先する（計算は行わない）。

出力先（既定）:
- `<run_dir>/relation_priors/relation_priors.json`

呼び出し:
- `simple_active_refine.relation_priors_compute.compute_and_save_relation_priors(...)`

主要引数:
- `dataset_dir`（train.txt を読む）
- `model_before_dir`（before KGE モデル。X7計算に必要）
- `RelationPriorConfig`（`--xr_*`）
  - `xr_weight_x2/x3/x4/x7`（統合重み。既定は `x7=1.0`）
  - `xr_min_count_x7`, `xr_max_samples_x3_per_relation`, `xr_max_samples_x7_per_relation`

prior の受け渡し規約（重要）:
- 優先順位: `--relation_priors_path`（明示） > Step0で計算した priors > arm-run 側の自動検出（`<dataset_dir>/relation_priors.json`）
- arm-run へは `relation_priors_path` として渡され、witness の重み付けに使われる

参照:
- relation priors の定義・運用: [docs/rules/RULE-20260118-RELATION_PRIORS-001.md](RULE-20260118-RELATION_PRIORS-001.md)

### 4.2 Step 1: 初期ルールプール生成

実行条件:
- `rule_pool_pkl` が存在し、`--force` が未指定なら **スキップ**。
- それ以外は生成。

出力先:
- `<run_dir>/rule_pool/`

呼び出し:
- `build_initial_rule_pool.build_initial_rule_pool(...)`

主な引数の受け渡し:
- `model_dir`（学習済みKGE）
- `target_relation`
- `output_dir = <run_dir>/rule_pool`
- ルール抽出設定（`n_rules`, `sort_by`, `mode`, `min_head_coverage`, `min_pca_conf`, `lower_percentile`, `k_neighbor`）

成功判定:
- 実行後に `<run_dir>/rule_pool/initial_rule_pool.pkl` が存在しない場合は例外で停止。

備考:
- ルール生成は LLM を使わず、AMIE+抽出ルールから初期プールを直接作る方針（実装に準拠）。

### 4.3 Step 2: 初期arm生成

実行条件:
- `arms_json` が存在し、`--force` が未指定なら **スキップ**。
- それ以外は生成。

出力先:
- `<run_dir>/arms/`

呼び出し:
- `build_initial_arms.build_initial_arm_pool(...)`

重要な挙動（実装準拠の注意点）:
- arm生成の入力として `candidate_triples`（典型: train_removed.txt）を渡す（既存実験互換）。
- 併せて `dir_triples=dataset_dir` を渡す。
  - これは pair-arm（ルールペア）選別で「support/Jaccard 計算母集団」を `train.txt` に切り替える場合に、`train.txt` を参照できるようにするため。

pair-arm support source（新規）:
- `pair_support_source=candidate`（既定）: pair-arm の support/Jaccard は candidate_triples を母集団に計算（後方互換）
- `pair_support_source=train`: pair-arm の support/Jaccard は `dataset_dir/train.txt` を母集団に計算（候補未知の問題設定に合わせたい場合）

ルール事前フィルタ（任意）:
- `--rule_filter_config` / `--exclude_relation_pattern` / `--rule_top_k` / `--arms_min_*` 等を通じて、singleton arm 数を抑えることを想定。

成功判定:
- 実行後に `<run_dir>/arms/initial_arms.json` が存在しない場合は例外で停止。

### 4.4 Step 3: arm反復精錬（arm_run）

実行条件:
- `<run_dir>/arm_run/` 配下に `iter_` で始まるディレクトリが1つでも存在し、`--force` が未指定なら **スキップ**。
- それ以外は実行。

出力先:
- `<run_dir>/arm_run/iter_k/`（k=1..n_iter）

呼び出し:
- `simple_active_refine.arm_pipeline.ArmDrivenKGRefinementPipeline.from_paths(...).run()`

入力パスの束縛（重要）:
- `initial_arms_path = <run_dir>/arms/initial_arms.json`
- `rule_pool_pkl = <run_dir>/rule_pool/initial_rule_pool.pkl`
- `dir_triples = dataset_dir`（この中の `train.txt` が現KGの初期としてロードされる）
- `target_triples_path = target_triples`
- `candidate_triples_path = candidate_triples`

補足（任意入力ファイル）:
- `dataset_dir/relation2text.txt` が存在する場合、arm選択（特に `selector_strategy=llm_policy`）のために
  ターゲット関係/ルールpredicateの自然言語説明として利用される。

主要パラメータ:
- `n_iter`, `k_sel`, `n_targets_per_arm`, `max_witness_per_head`
- selector: `selector_strategy`（`ucb|epsilon_greedy|llm_policy|random`）、`selector_exploration_param`, `selector_epsilon`
- reward: `witness_weight`, `evidence_weight`

relation priors（任意）:
- `relation_priors_path`: JSON mapping predicate->prior（または payload形式）
  - 指定が無い場合、arm-run 側が `<dataset_dir>/relation_priors.json` を自動検出する

KG更新ポリシー（実装のポイント）:
- 反復中にKGへ追加されるのは **evidence triples のみ**（accepted_evidence）。
- hypothesis triples は `pending_hypothesis_triples.tsv` に退避され、KGへ確定追加しない（store-only）。

incident triples（任意の拡張）:
- accepted evidence が新規エンティティを導入した場合に、candidate（例: train_removed）から incident triples を追加し、dangling entity を避ける。
- target score 重視の実験では、incident の追加量が大きいとノイズになり得るため、次のCLIで制御できる。
  - `--disable_incident_triples`: incident 追加を無効化
  - `--max_incident_candidate_triples_per_iteration K`: iteration あたりの incident 追加数を上限Kに制限

推奨（KG-FIT運用）:
- KG-FIT ではテキスト属性で新規エンティティをアンカーできる可能性があるため、
  まずは「incident なし（OFF）」と「小さめ上限（例: 0/50/200）」で target score の差分を確認してから本番運用へ進む。

### 4.5 Step 4: retrain + evaluate

実行条件:
- `<run_dir>/arm_run/retrain_eval/summary.json` が存在し、`--force` が未指定なら **スキップ**。
- それ以外は実行。

呼び出し:
- `retrain_and_evaluate_after_arm_run.run(...)`

入力:
- `run_dir = <run_dir>/arm_run`（arm-run出力。`iter_*/accepted_evidence_triples.tsv` を集約する）
- `dataset_dir`（元データセット: train/valid/test）
- `target_triples`
- `after_mode`: `load|retrain`
- `embedding_config`, `num_epochs`, `force_retrain`
- `exclude_predicate`（追加evidenceから除外するpredicate群）
- `model_before_dir`, `model_after_dir`（指定がなければ `arm_run/retrain_eval/model_before|model_after`）

重要: beforeモデルの取り扱い
- 評価工程は before を **必ず「学習済みモデルディレクトリ」からロード**し、ディレクトリが未整備だと停止する。
- 統合ランナーは `model_dir` を自動で `model_before_dir` にコピー/リンクしない。
- よって実運用では次のいずれかが必要:
  1) `--model_before_dir` を明示指定して `--model_dir` と同じディレクトリを渡す
  2) `<run_dir>/arm_run/retrain_eval/model_before` に事前にモデルを配置する

成功判定:
- 実行後に `<run_dir>/arm_run/retrain_eval/summary.json` が存在しない場合は例外で停止。

運用上の注意（early stopping / stopper）:
- `embedding_config` に `stopper="early"` を指定する場合は、validation triples が必要。
- `valid.txt` が空のデータでは stopper に起因して学習が例外停止する場合があるため、現行実装では `KnowledgeGraphEmbedding.train_model()` が
  validation が無いとき stopper を自動無効化して学習を継続する（詳細は retrain/eval 仕様を参照）。

---

## 5. 冪等性（スキップ）と上書き（--force）

### 5.1 スキップ条件（force=False）

- Step 1: `<run_dir>/rule_pool/initial_rule_pool.pkl` が存在 → skip
- Step 2: `<run_dir>/arms/initial_arms.json` が存在 → skip
- Step 3: `<run_dir>/arm_run/iter_*` が1つでも存在 → skip
- Step 4: `<run_dir>/arm_run/retrain_eval/summary.json` が存在 → skip

### 5.2 上書き条件（force=True）

`--force` 指定時、各ステップの出力ディレクトリが「存在し、かつ中身がある」場合は削除して作り直す。

- 対象ディレクトリ:
  - `rule_pool_dir`, `arms_dir`, `arm_run_dir`
- 動作:
  - `shutil.rmtree(dir)` で削除 → `mkdir(parents=True)`

注意:
- ステップ単位の `--start-from` / `--stop-after` は現行実装には存在しない。
- `--force` は「全工程の再生成」を意味しやすいので、部分的にやり直したい場合は成果物を手で削除する運用となる。

---

## 6. CLI仕様（main）

`main()` は argparse で CLI を提供し、`run_pipeline(...)` へ引数をそのまま中継する。

- rule pool系: `--n_rules`, `--sort_by`, `--mode {entire,high-score}`, `--min_head_coverage`, `--min_pca_conf`, `--lower_percentile`, `--k_neighbor`
- arms系: `--k_pairs`, `--pair_support_source {candidate,train}`, `--max_witness_per_head`, `--rule_top_k`, `--rule_sort_by`, `--arms_min_*`, `--exclude_relation_pattern`（append）, `--rule_filter_config`
- arm-run系: `--n_iter`, `--k_sel`, `--n_targets_per_arm`, `--selector_strategy`, `--selector_exploration_param`, `--selector_epsilon`, `--witness_weight`, `--evidence_weight`
- relation priors系: `--relation_priors_path`, `--compute_relation_priors`, `--relation_priors_out`, `--xr_*`
- eval系: `--after_mode {load,retrain}`, `--embedding_config`, `--num_epochs`, `--force_retrain`, `--model_before_dir`, `--model_after_dir`, `--exclude_predicate`（append）
- runner系: `--force`

---

## 7. 失敗時の挙動（例外/終了）

- スキップされない工程で期待成果物が生成されない場合、`RuntimeError` を投げて停止する。
- `--force` 未指定で出力ディレクトリが非空の場合、`SystemExit` で停止する（ディレクトリ上書きを防ぐ）。
- 評価工程で before/after モデルディレクトリが不正な場合、`SystemExit(2)` で停止する。

---

## 8. 実装参照（ソースコード）

- 統合ランナー本体: [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py)
- Step1（初期ルールプール）: [build_initial_rule_pool.py](../../build_initial_rule_pool.py)
- Step2（初期arm）: [build_initial_arms.py](../../build_initial_arms.py)
- Step3（arm-run）: [simple_active_refine/arm_pipeline.py](../../simple_active_refine/arm_pipeline.py)
- Step4（retrain/eval）: [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)

---

## 9. 参考文献（docs）

本仕様の背景・前提となる設計/記録:

- arm-based精錬の設計（witness/衝突、出力規約、after_mode 等）: [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- v3パイプラインのI/F整理（抽象I/Fの考え方）: [docs/rules/RULE-20260111-PIPELINE_OVERVIEW-001.md](RULE-20260111-PIPELINE_OVERVIEW-001.md)
- 統合ランナーの実装計画（本スクリプトの設計意図）: [docs/records/REC-20260114-ARM_PIPELINE-001.md](../records/REC-20260114-ARM_PIPELINE-001.md)
- 実験計画（run_full_arm_pipelineの使い方・期待出力例）: [docs/records/REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md](../records/REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md)
- モジュール責務一覧（全体像）: [Agents.md](../../Agents.md)
