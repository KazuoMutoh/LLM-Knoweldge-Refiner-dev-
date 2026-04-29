# RULE-20260201-KGE_RETRAIN_EVAL_PROTOCOL-001: arm-run 追加トリプルのKGE再学習評価プロトコル（incremental / cumulative / random baseline）

作成日: 2026-02-01
最終更新日: 2026-02-01

## 0. 目的

arm-run（反復精錬）で得た追加トリプルが、
- ターゲットトリプルのスコア（例: nationality）
- KGEのリンク予測性能（Hits@k, MRR）

に与える因果的な影響を、再現可能な条件で比較・評価するための標準手順を定義する。

本プロトコルの狙いは次の2点:
- **「どのiterationの追加が効いたか」**（増分）と **「累積でどこまで伸びるか」**（和集合）を分けて観測する
- 追加数の差による見かけの改善を避けるため、**同数のランダム追加**を対照条件として用意する

## 1. スコープ

- 評価スクリプト（実装）:
  - [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py)
  - 仕様: [docs/rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md](RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md)
- 想定データセット:
  - incident removal / head_incident_v1 等のテストデータセット（`dir_triples` が自己完結）
- 対象外:
  - arm選択やWeb取得の改善（それ自体は別の実験設計・別rules）

## 2. 用語

- **before**: 追加前の学習済みKGE（固定）
- **after**: 追加トリプルを train に加えた上で再学習したKGE
- **incremental（増分）**: ある時点の追加（例: iter=3 の accepted_added_triples のみ）で after を作る
- **cumulative union（累積和集合）**: iter=1..k の追加の union で after を作る
- **random baseline（同数ランダム）**: incremental/cumulative の追加数と同数だけ、候補集合からランダムに追加して after を作る

## 3. 入力の固定（比較可能性の必須条件）

- `dataset_dir`（train/valid/test + text）が同一
- `target_triples` が同一
- `model_before_dir` が同一
- `embedding_config` が同一（特に embedding_backend / model / seed / negative sampler 等）

注意:
- 評価の「比較」は、上記が揃って初めて意味がある。
- Web候補のように新規entityが入る場合も、比較軸（before/after）は同様。

## 4. 評価パターン

### 4.1 incremental（iter=k のみ）

目的:
- そのiterationの追加が効いているか（悪化させていないか）を切り出して見る。

手順（標準）:
1. 評価用に `run_dir` を「iter=k の追加だけを集約できる形」にする
   - 例: `run_dir_incremental_iter3/iter_1/accepted_added_triples.tsv` に iter3 のファイルをコピーして配置
   - `retrain_and_evaluate_after_arm_run.py` は `iter_*/accepted_added_triples.tsv` を全結合するため、run_dir を分けて入力を制御する
2. `retrain_and_evaluate_after_arm_run.py` を実行する

### 4.2 cumulative union（iter=1..k）

目的:
- 「累積で最終的にどこまで伸びるか」を観測する。

手順（標準）:
- arm-run の `run_dir` をそのまま渡す（デフォルトで iter 全結合の union 評価になる）

### 4.3 同数ランダム（random baseline）

目的:
- 追加数の差による改善を排除し、選択/取得の価値を評価する。

標準条件:
- 追加数 N は incremental/cumulative で得た `accepted_added_triples.tsv` のユニーク件数
- ランダム抽出元は、比較対象と整合する候補集合（例: train_removed、または同条件で作った候補プール）
- 同一seedで再現可能にする

実装メモ:
- ランダム条件の追加トリプル生成は、評価run_dirに `iter_1/accepted_added_triples.tsv` を「ランダムN件」で作ってから同じ評価スクリプトを回す、という運用で統一する（評価スクリプト自体にランダム生成は含めない）。

## 5. hypothesis predicate の取り扱い（leak防止）

原則:
- after 学習の train に **target predicate（例: `/people/person/nationality`）を混入させない**。

手段:
- `retrain_and_evaluate_after_arm_run.py` に `--exclude_predicate <target_predicate>` を渡す。

補足:
- evidence-first / store-only hypothesis の設計でも、incident triples や Web候補から混入する可能性があるため、評価時に明示的に除外する。

## 6. 実行手順（推奨の2段階）

1) sanity（軽量）:
- `--num_epochs 2`
- 目的: updated_triples 作成・after 学習・summary.json 出力が揃うことの確認

2) main（本番）:
- `--num_epochs 100`（または既定の学習設定に合わせる）

出力:
- `run_dir/retrain_eval/summary.json`
- `run_dir/retrain_eval/evaluation/`（Hits@k, MRR 等）

## 7. 参照（検討記録）

- LLM-policy（local候補）: [docs/records/REC-20260129-LLM_POLICY_NATIONALITY_TRIPLE_ACQUISITION-001.md](../records/REC-20260129-LLM_POLICY_NATIONALITY_TRIPLE_ACQUISITION-001.md)
- LLM-policy + Web候補（iter=25）: [docs/records/REC-20260201-LLM_POLICY_WEB_TRIPLE_ACQUISITION_NATIONALITY-001.md](../records/REC-20260201-LLM_POLICY_WEB_TRIPLE_ACQUISITION_NATIONALITY-001.md)
- iter3/final vs random の再学習評価計画: [docs/records/REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001.md](../records/REC-20260129-ITER3_FINAL_KGE_EVAL_NATIONALITY-001.md)
