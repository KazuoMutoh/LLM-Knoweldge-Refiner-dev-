# REC-20260118-FULL_PIPELINE_RESULTS-001: run_full_arm_pipeline 実行結果（2026-01-18）

作成日: 2026-01-18

参照:
- [REC-20260116-FULL_PIPELINE_EXPERIMENT-001](REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md): 実験設計（ベースライン定義）
- [KGEフレンドさを考慮したwitness評価の改善](../external/KGEフレンドさを考慮したwitness評価の改善.md): witness評価のprior導入
- [REC-20260118-ADD_ALL_TRAIN_REMOVED-001](REC-20260118-ADD_ALL_TRAIN_REMOVED-001.md): train_removed を全投入しても target score が悪化するベースライン検証
- [REC-20260117-ARM_SELECTOR-001](REC-20260117-ARM_SELECTOR-001.md): LLM-policy選択の改良方針
- [RULE-20260118-RELATION_PRIORS-001](../rules/RULE-20260118-RELATION_PRIORS-001.md): relation priors（X_r）とwitness重み付けの標準
- [RULE-20260117-ARM_SELECTOR_IMPL-001](../rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md): LLM-policy/UCBのarm選択I/Fと入力情報
- [RULE-20260117-ARM_PIPELINE_IMPL-001](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md): arm-run（witness/incident/accepted_added）の実装仕様
- [RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001](../rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md): accepted_added優先集約とretrain/eval仕様
- [RULE-20260117-RUN_FULL_ARM_PIPELINE-001](../rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md): 統合ランナー仕様（Step0: priors計算含む）

---

## 0. 目的

`run_full_arm_pipeline.py` による一連の実験について、**実行結果（コマンド・成果物・指標）**を別ファイルとして集約する。

---

## 1. REC-20260116 の設計から何が変わったか（差分）

本ファイルで扱う実行は、[REC-20260116-FULL_PIPELINE_EXPERIMENT-001](REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md) で設計した枠組みを踏襲しつつ、実装・運用の観点で次の差分が入っている。

### 1.1 witness評価の改良（relation priorsによる重み付け）

- [KGEフレンドさを考慮したwitness評価の改善](../external/KGEフレンドさを考慮したwitness評価の改善.md) の方針に沿って、relation prior $X_r$ を導入。
- proxy reward 計算に入る witness を、relation prior により減衰（raw witness は診断用に保持）。
- 実験により `--compute_relation_priors` / `--relation_priors_path` を使い、`relation_priors.json` を生成・投入。
  - 本日の実験では、デフォルト設定として $X_r = X_r(7)$（幾何的一貫性）を主に使用（$X_r(2..4)$ は重み0）。

参照（実装準拠の仕様）:
- [RULE-20260118-RELATION_PRIORS-001](../rules/RULE-20260118-RELATION_PRIORS-001.md)
- [RULE-20260117-ARM_PIPELINE_IMPL-001](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)
- [RULE-20260117-RUN_FULL_ARM_PIPELINE-001](../rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)

### 1.2 LLM-policy選択の改良（armの意味・取得/追加トリプル提示）

- [REC-20260117-ARM_SELECTOR-001](REC-20260117-ARM_SELECTOR-001.md) の方針に沿って、LLM-policy が参照できる情報を拡張。
  - armの意味（rule_keys/body predicates）
  - armが取得した evidence triples（既存KG含む）
  - armが新規に追加した evidence（accepted）
  - witness/coverage/overlap などの診断指標
  - 利用可能なら `relation2text.txt` 由来の relation description

参照（実装準拠の仕様）:
- [RULE-20260117-ARM_SELECTOR_IMPL-001](../rules/RULE-20260117-ARM_SELECTOR_IMPL-001.md)
- [RULE-20260117-ARM_PIPELINE_IMPL-001](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)

### 1.3 取得トリプルの拡張（?c を含む incident triples も追加対象）

- 追加トリプルの集約対象を `accepted_evidence_triples.tsv` から `accepted_added_triples.tsv` 優先に変更。
- `accepted_added_triples.tsv` は、head候補（例: `?a nationality ?b`）だけでなく、説明に必要な中間ノード `?c` を含む周辺トリプル（incident triples）も含み得る。
  - 例: `?a place_of_birth ?c` や `?c contained_by ?b` のような、候補生成で露出した中間エンティティ周りの候補トリプルを train に反映。

参照（実装準拠の仕様）:
- [RULE-20260117-ARM_PIPELINE_IMPL-001](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)
- [RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001](../rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md)

---

## 2. 共通設定（今回の実行群）

- dataset: `/app/experiments/test_data_for_nationality_v3`
- target relation: `/people/person/nationality`
- target_triples: 117
- test triples: 207（うち3がunknownでフィルタ）
- before model: `/app/models/20260116/fb15k237_transe_nationality`
- after training: `--num_epochs 100`（valid=0 のため early stopper は無効化）

---

## 3. 実験結果

### 3.1 exp_B4 rerun（LLM-policy / priors無し）

- run_dir: `/app/experiments/20260118/exp_B4_llm_25_rerun_20260118b`
- 条件: n_rules=20, k_pairs=20, n_iter=25, selector=llm_policy
- arm-run追加（accepted_added_triples）: 11195
- retrain+eval（test set）:
  - Target score: -9.8677 → -13.4426 (Δ=-3.5748)
  - Hits@1: 0.2426 → 0.2574 (Δ=+0.0147)
  - Hits@3: 0.5392 → 0.5319 (Δ=-0.0074)
  - Hits@10: 0.8235 → 0.7966 (Δ=-0.0270)
  - MRR: 0.4257 → 0.4322 (Δ=+0.0065)

### 3.2 arm10 + priors投入（LLM-policy）

- run_dir: `/app/experiments/20260118/exp_B4_llm_25_arm10_priors_20260118a`
- 条件: n_rules=10, k_pairs=0, n_iter=25, selector=llm_policy, priors=compute
- arm-run追加（accepted_added_triples）: 11747
- retrain+eval（test set）:
  - Target score: -9.8677 → -13.1763 (Δ=-3.3085)
  - Hits@1: 0.2426 → 0.2451 (Δ=+0.0025)
  - Hits@3: 0.5392 → 0.5000 (Δ=-0.0392)
  - Hits@10: 0.8235 → 0.8211 (Δ=-0.0025)
  - MRR: 0.4257 → 0.4230 (Δ=-0.0026)

### 3.3 arm10 + priors投入（UCB）

- run_dir: `/app/experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a`
- 条件: n_rules=10, k_pairs=0, n_iter=25, selector=ucb(c=1.0), priors=compute
- arm-run追加（accepted_added_triples）: 10185
- retrain+eval（test set）:
  - Target score: -9.8677 → -10.0441 (Δ=-0.1763)
  - Hits@1: 0.2426 → 0.2574 (Δ=+0.0147)
  - Hits@3: 0.5392 → 0.5343 (Δ=-0.0049)
  - Hits@10: 0.8235 → 0.8088 (Δ=-0.0147)
  - MRR: 0.4257 → 0.4336 (Δ=+0.0080)

---

## 4. まとめ（所見）

- priors投入＋arm数削減（10）でも、target score と Hits@k の改善は一貫しない。
- 一方で、arm10+priors+UCB は MRR/Hits@1 の改善が見られ、LLM-policyより安定している可能性がある。
- `valid.txt` が空で early stopping が効かない点は、比較解釈上の大きな交絡要因。

### 4.1 追加分析: target head（?a）近傍増分と target score

前回（REC-20260117-FULL_PIPELINE_RESULTS-001）で観測した「?a 周りのエンティティ増加が target score を押し上げる」仮説について、今回の3条件で簡易に検証した。

- 方法: `updated_triples/added_triples.tsv` から target head ごとの 1-hop/2-hop 近傍増分（追加トリプル数・追加近傍エンティティ数）を集計し、target triple score の変化と比較。
  - raw score はスケールが変動し得るため、各モデルの train score 分布（サンプル）で標準化した $z$ も併記。
- 結果（LLM-policy 2条件）:
  - per-head の平均 $\Delta z$ は負（rerun: -0.74、arm10+priors: -0.68）で、target が相対的に悪化している。
  - 近傍増分と $\Delta z$ の相関は一貫して負（例: 1-hop unique neighbors は rerun: -0.47、arm10+priors: -0.30）。
  - したがって本日の条件では「?a 近傍が増えるほど target が上がる」傾向は再現しなかった。
- 結果（UCB 1条件）:
  - per-head の平均 $\Delta z$ は正（+0.20）で、raw では微減でも target は相対的に改善している可能性がある。
  - ただし近傍増分と $\Delta z$ の相関は弱い負（1-hop unique neighbors: -0.16）で、改善の主因が「?a 近傍増分」とは言いにくい。

再現用スクリプトと集計結果:
- [tmp/debug/analyze_target_head_neighborhood.py](../../tmp/debug/analyze_target_head_neighborhood.py)
- rerun（LLM-policy）: [arm_run_summary.md](../../experiments/20260118/exp_B4_llm_25_rerun_20260118b/arm_run/retrain_eval/analysis_target_head/arm_run_summary.md)
- arm10+priors（LLM-policy）: [arm_run_summary.md](../../experiments/20260118/exp_B4_llm_25_arm10_priors_20260118a/arm_run/retrain_eval/analysis_target_head/arm_run_summary.md)
- arm10+priors（UCB）: [arm_run_summary.md](../../experiments/20260118/exp_B4_ucb_25_arm10_priors_20260118a/arm_run/retrain_eval/analysis_target_head/arm_run_summary.md)

---

## 更新履歴

- 2026-01-18: 新規作成。REC-20260116 の結果章（9章以降）を分離し、実装差分（prior/LLM-policy/?c incident triples）と当日実験の結果を集約。
