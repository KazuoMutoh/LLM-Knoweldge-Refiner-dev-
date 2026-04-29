# REC-20260124-ARM_BUILDER_TRAIN_JACCARD-001: pair-arm の Jaccard 計算を train.txt ベースに変更する実装計画

作成日: 2026-01-24  
最終更新日: 2026-01-24

参照:
- arm生成（Step2）: [build_initial_arms.py](../../build_initial_arms.py)
- arm生成ロジック: [simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py)
- 統合ランナー: [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py)
- arm-run（Step3）: [simple_active_refine/arm_pipeline.py](../../simple_active_refine/arm_pipeline.py)
- Web候補の導入計画/記録: [docs/records/REC-20260123-ARM_WEB_RETRIEVAL-001.md](REC-20260123-ARM_WEB_RETRIEVAL-001.md)

---

## 0. 目的（問題設定と整合させる）

現在の問題設定では「追加トリプル候補（train_removed / Web取得 triples）は**事前には分からない**」とする。

一方、現状の pair-arm 生成は、`candidate_triples`（典型: train_removed）に依存して
- 各ルールがどの target_triple を支持できるか（support set）
- support set 同士の Jaccard（交差/和集合）
を計算し、上位のルールペアを pair-arm として採用している。

このままだと、pair-arm の構造が「本来未知な候補集合」に依存してしまい、問題設定と矛盾する。

したがって、pair-arm 選別（Jaccard）の計算母集団を **train.txt（既知KG）に限定**し、
「既知KGで共起しやすい（同じ target を説明しやすい）ルールペア」を選ぶように仕様変更する。

---

## 1. 現状（変更前）

- [simple_active_refine/arm_builder.py](../../simple_active_refine/arm_builder.py) の `build_initial_arms(...)` は
  - `candidate_triples` から `TripleIndex` を作成
  - 各 rule×target_triple に対して `supports_head` / `count_witnesses_for_head` を評価し support set を作成
  - support set の Jaccard 上位を pair-arm に採用

- [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py) は Step2 を
  - `build_initial_arm_pool(..., candidate_triples_path=--candidate_triples, dir_triples=None)`
  で呼んでおり、実質的に train_removed を使って pair-arm を選別している。

---

## 2. 変更後仕様（目標）

- singleton-arm（各ルール単体）は現状どおり全ルール作成
- pair-arm（ルールペア選別）の Jaccard は **train.txt のみ**を用いて計算できるようにする
  - 具体的には「support set 構築に用いる TripleIndex」を train.txt 由来に切り替え可能にする
- ただし既存実験との互換のため、既定値は **candidate（train_removed 等）** を維持する
  - 実験設定が整い次第、train を既定に切り替えることを検討する
- Step3（arm-run）での witness / evidence 取得は、実験設定に応じて local/web を使う

期待効果:
- pair-arm が「既知KG構造」に基づくため、問題設定（候補未知）と整合
- 事前に train_removed を用意できない / Web前提でも、pair-arm を作れる

---

## 3. 設計案

### 案A（推奨）: pair-arm 用の support triple source を追加し、train/candidate を切替可能に

`ArmBuilderConfig` に次を追加する:
- `pair_support_source: str = "candidate"`（`candidate|train`）

`build_initial_arms(...)` の引数を拡張し、
- `candidate_triples`（従来通りの引数名）は温存
- ただし pair-arm の support set 計算に使う triple 集合は、設定により
  - `train_triples`（train.txt）
  - `candidate_triples`（互換用）
 から選べるようにする

呼び出し側（Step2: build_initial_arm_pool）では
- `pair_support_source=train` のとき、`dir_triples/train.txt`（または明示ファイル）を読み、pair用 support triples として渡す
- 既定（candidate）のときは、従来通り `candidate_triples`（典型: train_removed）で pair を計算する

### 案B（最小変更）: build_initial_arms に train_triples を別引数で渡す

`build_initial_arms(..., pair_support_triples=...)` のような引数を増やして切り替える。

案Aより分かりやすいが、CLI/設定の一貫性（configで統制）を考えると案Aを推奨。

---

## 4. I/F と CLI 変更（提案）

### 4.1 `build_initial_arms.py`（CLI）
- `--dir-triples` を指定した場合、pair-arm の Jaccard は **train.txt** で計算する（標準）
- 追加オプション（必要なら）:
  - `--pair-support-source {train,candidate}`（デフォルト train）

互換:
- 既存運用（candidate_triples_path のみ指定）を壊さないため、
  `--dir-triples` 未指定の場合は
  - `pair_support_source=candidate` にフォールバックする（または明示エラー）

### 4.2 `run_full_arm_pipeline.py`（統合ランナー）
- Step2 は原則 `dir_triples=dataset_dir` を渡すように変更し、train.txt を必須にする
- `--candidate_triples` は
  - Step2 では「pair-arm 選別には不要」
  - ただし Step3（candidate_source=local）や incident triples のために必要
 という位置付けに整理する

※将来的に candidate_source=web をデフォルトに寄せる場合、Step2 は train だけで成立するようになる。

---

## 5. 実装ステップ（DoD付き）

### Step 1: arm_builder の拡張
- `ArmBuilderConfig` に `pair_support_source` を追加
- support set 構築時に、pair-arm 用 triple 集合を切り替える

DoD:
- 既存の設定（candidateベース）も選べる
- trainベース設定で pair-arm が生成される

### Step 2: build_initial_arms.py の配線変更
- `--dir-triples` 指定時は train.txt を読み込み、pair-arm 選別の母集団として使う

DoD:
- `--dir-triples` だけで pair-arm を作れる（candidate未指定でも）

### Step 3: run_full_arm_pipeline.py の Step2 を更新
- Step2 呼び出しを `dir_triples=dataset_dir` ベースに変更（pair-armはtrainで計算）

DoD:
- 従来の統合ランナー実験が、デフォルト挙動で動作する
- （差分として）pair-arm の中身が train 依存に変わることをログで確認できる

### Step 4: テスト追加/更新
- unit: `build_initial_arms` に対して
  - trainベース support set で pair-arm が作られる
  - candidateベースも後方互換として動く
- integration: `tests/test_run_full_arm_pipeline.py` に、Step2 の呼び出し引数が想定どおりになっていることを追加検証（モックで可）

---

## 6. リスクと対策

- trainだけだと rule body を満たす witness が減り、pair-arm の共起が弱くなる可能性
  - 対策: `k_pairs` を増やす、または `pair_support_source=candidate` を互換として残して比較できるようにする

- Web前提（candidate_source=web）の場合、Step3 の候補集合と Step2 の pair-arm 選別が別母集団になる
  - ただし本仕様変更の目的は「候補未知」という問題設定への整合であり、設計として許容

---

## 更新履歴
- 2026-01-24: 新規作成。pair-arm の Jaccard を train.txt ベースに切替可能にする方針と実装手順を整理。
- 2026-01-24: 既存実験互換のため、既定値は candidate（train_removed 等）を維持する旨を追記。
