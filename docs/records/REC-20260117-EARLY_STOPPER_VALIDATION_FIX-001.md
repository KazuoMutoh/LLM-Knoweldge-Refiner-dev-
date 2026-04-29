# REC-20260117-EARLY_STOPPER_VALIDATION_FIX-001: Early Stopper + No Validation Triples Crash Fix

作成日: 2026-01-17
最終更新日: 2026-01-17

## 背景

Series B/C 実験では `config_embeddings_with_stopper.json`（PyKEEN early stopping）を使用する。ところが本データ（`test_data_for_nationality_v3`）では `valid.txt` が空であり、retrain/eval の KGE 学習中に PyKEEN の early stopping が `evaluation_triples_factory=None` を参照して例外で停止した。

- 例外（抜粋）: `AttributeError: 'NoneType' object has no attribute 'mapped_triples'`
- 発生箇所: `simple_active_refine/embedding.py` → `KnowledgeGraphEmbedding.train_model()` → `pykeen.pipeline.pipeline(..., stopper='early', validation=None)`

このため、`exp_B1_ucb_25` の retrain/eval が途中で落ち、runner 全体が停止した。

## 方針

- 早期停止は validation triple が存在することが前提のため、validation が存在しない場合は stopper を無効化して学習を継続する。
- 既存データセットのファイル構造（valid/test が空でも動作）を維持し、最小修正で実験を前進させる。

## 実装

- `simple_active_refine/embedding.py` の `KnowledgeGraphEmbedding.train_model()` にガードを追加。
  - `pipeline_kwargs['stopper']` が設定されていて、かつ `valid_tf is None` の場合:
    - warning を出す
    - `stopper=None` にし、`stopper_kwargs` を取り除く

これにより、`config_embeddings_with_stopper.json` を指定しても valid が無いデータではクラッシュせずに学習が進む。

## 検証

- `exp_B1_ucb_25` を再開し、`run.log` に以下の warning が出たことを確認し、その後 training が継続することを確認:
  - `Stopper is configured (stopper='early') but validation triples are missing. Disabling stopper to avoid training crash.`

## 影響範囲

- validation が存在するデータ（built-in dataset を含む）では stopper は従来通り有効。
- validation が無いデータでは early stopping は無効化される（= 指定 epoch まで学習）。

## 関連

- 設計: [docs/rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md](../rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md)
- 実験計画: [docs/records/REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md](REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md)
