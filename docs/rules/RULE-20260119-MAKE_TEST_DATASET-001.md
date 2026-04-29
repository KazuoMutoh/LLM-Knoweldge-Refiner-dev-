# RULE-20260119-MAKE_TEST_DATASET-001: make_test_dataset.py テストデータセット生成 仕様

作成日: 2026-01-19

## 目的

- `make_test_dataset.py` による「部分的な文脈削除」テストデータセット生成の仕様を定義し、再現性を担保する。
- 生成物（`experiments/...`）を `run_full_arm_pipeline.py` 等の入力として利用できることを保証する。

## 入力

- `--dir_triples`: 元KGの triples ディレクトリ（例: `data/FB15k-237`）
- `--dir_test_triples`: 出力先（例: `experiments/test_data_for_nationality_v3`）
- `--target_relation`: ターゲットリレーション（例: `/people/person/nationality`）
- `--config`: 設定JSON（例: `config_dataset.json` または 既存生成物の `config_dataset.json`）

### 設定JSON（主要キー）

- `base_triples`: `train` / `valid` / `test`
- `target_entities`: `-`（auto）またはファイル/カンマ区切りリスト
- `auto_target_entities`: auto選択するターゲット数
- `min_target_triples`: auto選択時の「文脈が薄すぎるターゲット除外」閾値（0で無効）
- `target_preference`: `head` / `tail`
- `remove_preference`: `head` / `tail` / `both`（非target relationでの近傍方向）
- `drop_ratio`: 近傍エンティティのサンプリング率 $\rho \in [0,1]$
- `include_target`: 選択近傍エンティティの incident triples から target relation も削除対象に含めるか
- `seed`: 乱数seed

### 後方互換（読み替え）

- `manifest` は `manifest_filename` として扱う
- `selected_target_entities_filename` は `selected_target_entities_file` として扱う

## アルゴリズム概要

1. 元KGを読み込み、`base_triples` をベース集合 $G_B$ とする
2. ターゲットエンティティ集合 $T$ を決める
   - `target_entities != '-'` の場合: 指定を使用
   - `target_entities == '-'` の場合: `auto_target_entities` に従い、`target_relation` の triple から `target_preference` 側の一意エンティティを選ぶ
   - `min_target_triples > 0` の場合: 文脈が薄い候補（incident triples数が閾値未満）を除外
3. 近傍候補 $N$（neighbors_all）を集める
   - `target_relation` 以外の relation を経由した近傍エンティティ
   - `remove_preference` で in/out/both を切替
4. 近傍サンプリング: $R \subseteq N$（neighbors_selected）
   - $|R| \approx \rho |N|$
5. 削除集合 $D$ を構築
   - $D = \{(h,r,t)\in G_B \mid h\in R \lor t\in R,\; (include\_target \lor r\neq target\_relation)\}$
6. 出力
   - `train.txt` 等: $G_B \setminus D$（と他splitの同様処理）
   - `train_removed.txt` 等: $D \cap G_B$（と他splitの同様処理）
   - `target_triples.txt`: `target_relation` かつ `target_preference` 側が $T$ の triple
   - `selected_target_entities.txt`: $T$
   - `entity_removed_mapping.json`: target entity → 削除された triple の一覧
   - `config_dataset.json`: 設定と統計（加えて `target_entities_selected` / `neighbors_selected` 等を含み得る）

## 再現性（重要）

### 1) manifest からの厳密再現

既存生成物の `config_dataset.json` に以下が含まれる場合、`make_test_dataset.py` はそれらを優先して使用する。

- `target_entities_selected`
- `neighbors_selected`（必要なら `neighbors_all`）

条件:
- `target_entities` が `-`（auto）の場合

目的:
- 過去生成物（例: `experiments/test_data_for_nationality_v3`）と同一内容のデータを再生成可能にする。

### 2) 出力順序

- 出力ファイルの行順は、原則として **入力ファイル順（`dir_triples/train.txt` の行順）を保持**する。
- 過去生成物が異なる順序で書き出されている場合があるため、順序の厳密一致が必要なケースでは、別途「順序規約」または「ソート規約」を明示し、オプションとして実装する。

## 参照

- 実装: `make_test_dataset.py`
- 入力例: `config_dataset.json`
- 生成物例: `experiments/test_data_for_nationality_v3/`
