# REC-20260119-MAKE_TEST_DATASET_FIX-001: make_test_dataset.py のv3データ再現性（manifest対応）修正

作成日: 2026-01-19

## 目的

- `make_test_dataset.py` が最新の仕様・運用（`experiments/test_data_for_nationality_v3` など）と一致しているか確認する。
- 一致していない場合、実装を修正し、既存の `config_dataset.json`（manifest）から **同一内容のテストデータ**を再生成できるようにする。

## 背景

- `experiments/test_data_for_nationality_v3/config_dataset.json` は、生成時の設定に加え `target_entities_selected` / `neighbors_selected` などの **再現に必要な決定済みリスト**を含む。
- しかし従来の `make_test_dataset.py` は
  - CLI/設定キーの不整合（`manifest` vs `manifest_filename`、`selected_target_entities_filename` など）
  - デバッグ出力の混入
  - 既存manifestに含まれる決定済みリストを参照しない
  ため、v3データの再現に失敗し得た。

## 対応内容（実装）

- `make_test_dataset.py` を修正し、以下を満たすようにした。
  - 設定ファイルに存在する旧キーを新キーへ **後方互換**で読み替え
    - `manifest` → `manifest_filename`
    - `selected_target_entities_filename` → `selected_target_entities_file`
  - 必要なCLI引数（`base_triples`, `drop_ratio`, `include_target`, `seed`, `min_target_triples` 等）を明示的に追加
  - `config_dataset.json`（manifest）に `target_entities_selected` / `neighbors_selected` が存在し、かつ `target_entities` が `-`（auto）指定の場合は、
    **それらを優先して使用**して厳密再現できるようにした。

## 検証

### 検証1: 既存manifestから再生成（v3再現）

実行:

```bash
cd /app
rm -rf /app/tmp/debug/repro_from_manifest_v3
python3 make_test_dataset.py \
  --dir_triples /app/data/FB15k-237 \
  --dir_test_triples /app/tmp/debug/repro_from_manifest_v3 \
  --target_relation /people/person/nationality \
  --config /app/experiments/test_data_for_nationality_v3/config_dataset.json
```

結果:
- 生成ログで `removed=94226 out=177889` が既存v3と一致。
- `selected_target_entities.txt` / `target_triples.txt` は **md5一致**。
- `train.txt` / `train_removed.txt` は **内容（集合）が一致**（行の並びは一致しないが、含まれるトリプル集合は一致）。
  - 既存v3は過去生成物のため、出力順が「入力順（source train.txt順）」と一致しない可能性がある。
  - 本修正では、再現可能性と安定性のため **入力ファイル順を保持**する。

### 検証2: config_dataset.json（ルート）での生成

`/app/config_dataset.json` は旧キーを含むため、後方互換読み替えが動作することを確認した。

## 生成物

- 修正対象: `make_test_dataset.py`
- 検証用出力（作業ディレクトリ）:
  - `/app/tmp/debug/repro_from_manifest_v3/`
  - `/app/tmp/debug/repro_test_data_for_nationality_v3/`

## 次のアクション

- 出力順序を厳密一致させる必要が出た場合は、既存データ生成時の順序規約（またはソート規約）を `docs/rules` に追加し、`make_test_dataset.py` に明示オプションとして実装する。
