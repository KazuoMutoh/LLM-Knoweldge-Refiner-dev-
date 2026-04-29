# REC-20260117-WITNESS_EVAL-001: KGEフレンドさ（relation prior）を考慮したwitness評価の実装計画

作成日: 2026-01-17
最終更新日: 2026-01-18

## 目的

[witness評価改善の検討書](../external/KGEフレンドさを考慮したwitness評価の改善.md) に基づき、arm反復精錬におけるwitness由来の報酬が「ハブ/無意味関係によるwitness水増し」に過度に引っ張られないようにする。

本計画では、witnessカウント（置換数）自体は保持しつつ、報酬計算で用いるwitness寄与を relation prior（KGE-friendlyスコア）で重み付けする。

## スコープ

- 対象: arm反復精錬の proxy reward（witness + evidence）
- 追加: relation prior をJSONから読み込む仕組み
- 対象（追記）:
  - relation prior（X_r(2)〜X_r(4), X_r(7)）の算出スクリプトを実装し、`relation_priors.json` を生成できるようにする
  - 統合ランナー（run_full_arm_pipeline.py）から prior 計算を任意で実行できるようにする
- 非対象（今回実装しない）:
  - 初期arm生成のseedスコア（cooc×prior）反映（検討書「使い方2」）

## 仕様（検討書からの対応）

- ルール $h$ の body predicates を $(r_1, r_2, ...)$ とし、ルール重みを

  $$W(h)=\prod_i X_{r_i}$$

  とする（body=2 が主だが、実装は一般化して積で扱う）。

- あるtarget head $(x,r_*,y)$ に対する witness count $c_h(x,y)$ を、報酬に入れるときは

  $$\sum_h W(h)\,c_h(x,y)$$

  を用いる（現行実装の「ruleごとのwitnessをarm内で合算」構造に合わせ、arm×target単位のスコアへ集約）。

- witness count（整数）は従来どおり記録し、説明可能性（coverageやtop witnessed targets表示）を壊さない。

## 実装方針

### 1) 入力（relation priors JSON）

- JSON形式: `predicate -> prior`（数値）または `predicate -> {"X": prior, ...}` を許容
  - 追記: `{"meta": {...}, "priors": {...}}` のpayload形式（prior算出スクリプトの出力）も許容し、`priors` を自動でunwrapする
- priorは $[0,1]$ にclampする
- 既定探索:
  - 明示指定があればそれを使用: `--relation_priors_path`
  - 無ければ `<dataset_dir>/relation_priors.json` を自動検出

### 2) 変更箇所

- [simple_active_refine/arm_triple_acquirer_impl.py](../../simple_active_refine/arm_triple_acquirer_impl.py)
  - `ArmAcquisitionResult` に `witness_score_by_arm_and_target`（float）を追加
  - `LocalArmTripleAcquirer` に `relation_priors` を注入し、ルール重み $W(h)$ を計算
  - `witness_by_arm_and_target`（int）は従来どおり保持

- [simple_active_refine/arm_triple_evaluator_impl.py](../../simple_active_refine/arm_triple_evaluator_impl.py)
  - reward計算で `witness_score_by_arm_and_target` があればそれを使用
  - diagnostics に `witness_total_raw` を追加し、互換性のため `witness_total` は「rewardに使った合計」を維持

- [simple_active_refine/arm_pipeline.py](../../simple_active_refine/arm_pipeline.py)
  - `ArmPipelineConfig` に `relation_priors_path` を追加
  - `from_paths()` でprior JSONを読み込み、acquirerへ渡す
  - arm_historyのdiagnosticsに `witness_score_sum` / `mean_witness_score_per_target` を追加

- [run_arm_refinement.py](../../run_arm_refinement.py), [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py)
  - `--relation_priors_path` を追加し、configへ伝播

- 新規ユーティリティ:
  - [simple_active_refine/relation_priors.py](../../simple_active_refine/relation_priors.py)

### 2b) relation prior（X_r）算出の実装（追記）

- 新規: [simple_active_refine/relation_priors_compute.py](../../simple_active_refine/relation_priors_compute.py)
  - train triples から X2/X3/X4 を算出
  - beforeモデル（PyKEEN保存ディレクトリ）から entity embedding を読み込み X7 を算出
  - X = weighted sum（利用可能な成分のみでweightを正規化）として統合

- 新規CLI: [scripts/compute_relation_priors.py](../../scripts/compute_relation_priors.py)
  - `--dataset_dir` と `--model_before_dir` を受け取り `relation_priors.json` を生成

- 統合ランナー: [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py)
  - `--compute_relation_priors` を指定した場合、arm-run前に prior を計算して `ArmPipelineConfig.relation_priors_path` に渡す

### 3) 互換性

- 既存の `witness_by_target`（ArmHistory/LLM-policy提示）や `targets_with_witness` は raw witness に基づくため、従来どおり動作する。
- rewardのみが「priorで減衰されたwitness寄与」に置き換わる。

## テスト計画

- 既存 smoke に加え、priorを与えた場合に
  - raw witness count は変わらない（例: 2）
  - reward が `evidence_count + weighted_witness_score` になる

を検証するユニットテストを追加する。

- 追記: prior算出モジュールのユニットテスト
  - KGE無しでも X2/X3/X4→X が計算できる
  - ダミーKGEを与えると X7 が計算される

## 運用・移行

- priorファイルが無い環境では、従来と同じ挙動（重み=1.0）となる。
- prior導入後は、rewardスケールが変わる（witness寄与が0〜小さくなる）ため、必要に応じて
  - `--witness_weight` の再調整
  - evidenceとの比重調整

を行う。

## 関連

- 検討書: [docs/external/KGEフレンドさを考慮したwitness評価の改善.md](../external/KGEフレンドさを考慮したwitness評価の改善.md)
