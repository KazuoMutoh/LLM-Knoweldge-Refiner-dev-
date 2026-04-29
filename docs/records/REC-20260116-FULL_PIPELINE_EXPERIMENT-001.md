# REC-20260116-FULL_PIPELINE_EXPERIMENT-001: 本格的実験計画（run_full_arm_pipeline.py）

作成日: 2026-01-16
参照: 
- [RULE-20260111-COMBO_BANDIT_OVERVIEW-001](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- [REC-20260114-ARM_PIPELINE-001](REC-20260114-ARM_PIPELINE-001.md)
- [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py)

---

## 0. 実験の目的

本実験では、`run_full_arm_pipeline.py`を用いて、arm-based knowledge graph refinementの**本格的な実証実験**を実施する。

### 目的
1. **スケーラビリティの検証**: FB15k-237全体で初期ルールプール→arm生成→反復精錬→再学習・評価の全パイプラインが実行可能であることを確認
2. **精度改善の定量評価**: ターゲット関係（`/people/person/nationality`）に対するHits@k、MRR、target scoreの改善を測定
3. **パラメータ感度の分析**: arm選択戦略、witness/evidence重み、反復回数などの影響を把握
4. **計算資源の見積もり**: 実験実行時間、メモリ使用量、APIコールなどを記録し、今後の実験設計の基礎データとする

---

## 1. 実験設定

### 1.1 データセット
- **データセット名**: FB15k-237 (test_data_for_nationality_v3サブセット)
- **パス**: `/app/experiments/test_data_for_nationality_v3`
- **元データセット**: `/app/data/FB15k-237`
- **ターゲット関係**: `/people/person/nationality`
- **理由**: 
  - test_data_for_nationality_v3はnationality関係に特化した実験用データセット
  - すでにtarget_triples.txt、train_removed.txtなどが準備済み
  - nationalityは機能的（1対1寄り）であり、衝突検出の検証に適している
  - 前回の小規模実験で基本動作を確認済み

### 1.2 事前準備
実験実行前に以下を準備する必要がある：

#### 1) 学習済みKGEモデル（before）
- **目的**: 初期ルールプール抽出とbefore評価に使用
- **必要ファイル**:
  - `trained_model.pkl`
  - `metadata.json`
  - `training.pt` (optional)
- **推奨ディレクトリ**: `/app/models/20260116/fb15k237_transe_nationality`

#### 2) ターゲットトリプルリスト（target_triples.txt）
- **形式**: TSV (head \t relation \t tail)
- **内容**: `/people/person/nationality`のトリプル（**trainセットから抽出**）
- **既存ファイル**: `/app/experiments/test_data_for_nationality_v3/target_triples.txt`
- **注**: すでに準備済みのため、新規生成は不要

#### 3) 候補トリプル（train_removed.txt）
- **形式**: TSV
- **内容**: trainセットから除外されたトリプル
- **既存ファイル**: `/app/experiments/test_data_for_nationality_v3/train_removed.txt`
- **目的**: arm適用時の候補集合として使用
- **注**: すでに準備済みのため、新規生成は不要

#### 4) 設定ファイル
- `config_embeddings.json`: KGEモデルのハイパーパラメータ
  - `stopper`: early stoppingの設定（推奨: `"stopper": "early"` でvalidation lossの改善が止まったら自動停止）
- `config_rule_filter.json`: ルールフィルタ設定（optional）

---

## 2. 実験シリーズ

### 実験A: ベースライン実験（小規模・動作確認）
**目的**: パイプライン全体の動作確認と実行時間の見積もり

| パラメータ | 値 | 理由 |
|:---|:---|:---|
| n_rules | 20 | 初期ルールプールのサイズ |
| k_pairs | 10 | pair-armの上位数（小規模で動作確認） |
| n_iter | 10 | 反復回数（小規模で傾向把握） |
| k_sel | 3 | 各反復で選択するarm数 |
| n_targets_per_arm | 20 | 各armあたりの対象トリプル数 |
| selector_strategy | ucb | UCB（探索と活用のバランス） |
| selector_exploration_param | 1.0 | UCBの探索パラメータ |
| witness_weight | 1.0 | witness報酬の重み |
| evidence_weight | 1.0 | evidence報酬の重み |
| after_mode | retrain | トリプル追加後に再学習を実施 |
| num_epochs | 10 | 再学習時のエポック数 |

**実行コマンド例**:
```bash
python run_full_arm_pipeline.py \
  --run_dir experiments/20260116/exp_A_baseline \
  --model_dir models/20260116/fb15k237_transe_nationality \
  --target_relation /people/person/nationality \
  --dataset_dir experiments/test_data_for_nationality_v3 \
  --target_triples experiments/test_data_for_nationality_v3/target_triples.txt \
  --candidate_triples experiments/test_data_for_nationality_v3/train_removed.txt \
  --n_rules 20 \
  --k_pairs 10 \
  --n_iter 10 \
  --k_sel 3 \
  --n_targets_per_arm 20 \
  --selector_strategy ucb \
  --after_mode retrain \
  --num_epochs 10
```

**期待される出力**:
- `exp_A_baseline/rule_pool/initial_rule_pool.pkl`
- `exp_A_baseline/arms/initial_arms.json`
- `exp_A_baseline/arm_run/iter_k/` (k=1..10)
  - `selected_arms.json`
  - `accepted_evidence_triples.tsv`
  - `pending_hypothesis_triples.tsv`
  - `arm_history.json`
  - `diagnostics.json`
- `exp_A_baseline/arm_run/retrain_eval/summary.json`

**評価指標**:
- 実行時間（ステップごと）
- 追加されたevidence数
- arm選択の分布
- before/afterのHits@k、MRR、target score

---

### 実験B: 選択戦略と反復回数の組み合わせ
**目的**: arm選択戦略と反復回数の影響を検証

| 実験ID | selector_strategy | n_iter | 説明 |
|:---|:---|:---|:---|
| exp_B1 | ucb | 25 | UCB × 短期反復 |
| exp_B2 | ucb | 50 | UCB × 中期反復 |
| exp_B3 | ucb | 100 | UCB × 長期反復 |
| exp_B4 | llm_policy | 25 | LLM-policy × 短期反復 |
| exp_B5 | llm_policy | 50 | LLM-policy × 中期反復 |
| exp_B6 | random | 25 | Random × 短期反復（ベースライン） |
| exp_B7 | random | 50 | Random × 中期反復（ベースライン） |
| exp_B8 | random | 100 | Random × 長期反復（ベースライン） |

---

### 実験C: witness/evidence重みチューニング
**目的**: 報酬関数の重みパラメータの影響を検証

**報酬関数の定義**:
各armの報酬は以下の式で計算される：
```
reward = witness_weight × witness_sum + evidence_weight × accepted_count
```
ここで：
- `witness_sum`: そのarmで各targetに対するwitnessの合計（bodyパターンが成立した置換の数）
- `accepted_count`: 受理されたevidenceトリプルの数（既存KGに含まれていない新規トリプル）
- `witness_weight`: witness報酬の重み（デフォルト=1.0）
- `evidence_weight`: evidence報酬の重み（デフォルト=1.0）

**重みの意味**:
- **witness_weight**: bodyパターンの「説明の厚み」をどれだけ重視するか
  - 高くすると、多くのwitnessを持つ（＝より強く支持される）candidateを生成するarmが優遇される
- **evidence_weight**: 実際に追加されたトリプルの「数」をどれだけ重視するか
  - 高くすると、多くの新規トリプルを追加できるarmが優遇される

| 実験ID | witness_weight | evidence_weight | 説明 |
|:---|:---|:---|:---|
| exp_C1 | 1.0 | 1.0 | バランス（ベースライン） |
| exp_C2 | 2.0 | 1.0 | witness重視 |
| exp_C3 | 1.0 | 2.0 | evidence重視 |
| exp_C4 | 0.5 | 1.5 | evidence強調 |

**固定パラメータ**:
- n_rules=20, k_pairs=20, n_iter=20, k_sel=3, n_targets_per_arm=20
- selector_strategy=ucb
- num_epochs=100（early stopping有効）
- その他は実験Aと同様

**比較観点**:
- 報酬の構成（witness vs evidence）
- arm選択の傾向
- 精度への影響

#### 仮説：witness重視 vs evidence重視の効果

**仮説1: witness重視（exp_C2: witness=2.0, evidence=1.0）の効果**

*期待される振る舞い*:
- 複数の根拠（witness）で支持される候補を優先するarmが選ばれる
- 追加されるトリプル数は少なめだが、**質が高い**（複数のルールで一貫して支持される）
- ノイズ（低信頼度の候補）が混入しにくい

*KGE精度への影響*:
- **Target scoreの改善が大きい**: 対象トリプルが強く支持される構造を持つため、埋込空間での配置が改善される
- **Hits@1の向上**: 高信頼度候補が上位にランクされやすい
- **MRRの向上**: 正解トリプルのランクが上がる
- ただし、**追加トリプル数が少ない**ため、全体的なカバレッジ拡大は限定的

*想定されるリスク*:
- 高witness候補が偏在する（人気エンティティ周辺に集中）場合、多様性が不足
- hub bias（高次数ノード経由で水増しされたwitness）により、見かけ上のスコアが高くなる可能性

---

**仮説2: evidence重視（exp_C3: witness=1.0, evidence=2.0）の効果**

*期待される振る舞い*:
- 多くの新規トリプルを追加できるarmが選ばれる
- 追加されるトリプル数は多いが、**witness数が少ない候補も含まれる**（根拠が弱い）
- より広範囲の知識グラフ領域にトリプルが追加される

*KGE精度への影響*:
- **全体的なHits@3/10の向上**: 知識グラフの構造が豊かになり、多様なパスが利用可能になる
- **カバレッジの拡大**: より多くのエンティティ・関係の組み合わせがカバーされる
- **Target scoreの改善は中程度**: 個別トリプルの支持構造は弱いが、全体の文脈情報が増える

*想定されるリスク*:
- ノイズの混入: 根拠が弱い候補も追加されるため、誤ったトリプルが含まれる可能性
- 学習の不安定化: 低品質トリプルにより、埋込学習が悪影響を受ける可能性
- 計算コスト増: 追加トリプル数が多いため、再学習の時間が長くなる

---

**仮説3: evidence強調（exp_C4: witness=0.5, evidence=1.5）の効果**

*期待される振る舞い*:
- exp_E3よりさらにevidence重視（witnessの影響を相対的に低減）
- witnessが少なくても、多くのトリプルを追加できるarmが優先される
- **探索的（exploratory）**な振る舞い：新規領域への拡張

*KGE精度への影響*:
- **発見的効果**: これまで接続が薄かった領域に新しいパスが開かれる可能性
- **長期的な改善**: 初期は精度向上が小さくても、後のイテレーションで効いてくる可能性
- **リスク大**: ノイズ混入の可能性がさらに高まる

---

**仮説4: バランス型（exp_C1: witness=1.0, evidence=1.0）の効果**

*期待される振る舞い*:
- witnessとevidenceを等しく評価
- **質と量のトレードオフ**を自然にバランス

*KGE精度への影響*:
- **安定した改善**: 極端な偏りがないため、リスクが低い
- **中庸的な結果**: 最高ではないが、安全な選択肢

---

#### 実験Cで検証すべき項目

1. **追加トリプル数の推移**:
   - 各設定でのiter毎の追加数を比較
   - witness重視 < バランス < evidence重視 の関係が成立するか

2. **witness分布**:
   - 追加されたトリプルのwitness数のヒストグラム
   - witness重視では高witness候補が多いか

3. **精度指標の変化**:
   - Hits@1: witness重視が有利か？
   - Hits@3/10: evidence重視が有利か？
   - MRR: どちらが有利か？
   - Target score: witness重視が有利か？

4. **arm選択の多様性**:
   - エントロピーやGini係数で選択の偏りを測定
   - evidence重視ではより多様なarmが選ばれるか

5. **追加トリプルの質**:
   - 追加後のKGでの接続性（次数、クラスタリング係数など）
   - witness重視では構造的に重要なトリプルが追加されるか

6. **計算時間**:
   - evidence重視では追加数が多いため、再学習時間が長くなるか

7. **相対改善率**:
   - 追加トリプル数あたりの精度改善（効率性の指標）
   - witness重視では効率が高いか

---

**予想される結果（検証すべき仮説）**:

| 指標 | witness重視(C2) | バランス(C1) | evidence重視(C3) | evidence強調(C4) |
|:---|:---:|:---:|:---:|:---:|
| 追加トリプル数 | 少 | 中 | 多 | 最多 |
| Hits@1改善 | 大 | 中 | 中 | 小 |
| Hits@10改善 | 中 | 中 | 大 | 大 |
| MRR改善 | 大 | 中 | 中 | 小 |
| Target score改善 | 最大 | 中 | 中 | 小 |
| 追加トリプル平均witness | 高 | 中 | 低 | 最低 |
| 計算時間（再学習） | 短 | 中 | 長 | 最長 |
| 効率性（改善/追加数） | 高 | 中 | 低 | 最低 |

**推奨される選択**（事前予想）:
- **Target scoreの改善が最優先**: witness重視（C2）
- **全体的な精度改善**: バランス型（C1）または軽度のwitness重視
- **カバレッジ拡大が目的**: evidence重視（C3）
- **計算資源に余裕がある**: evidence重視で多くのトリプルを追加し、長期的な改善を狙う

---

## 3. 実験実行計画

### 3.1 実行順序
1. **準備フェーズ**（手動）:
   - KGEモデルの学習（別スクリプト）
   - target_triples.txtの生成
   - candidate_triples.txtの準備

2. **実験A**（最優先）:
   - パイプライン動作確認
   - 実行時間の見積もり
   - 出力の検証

3. **実験B/C**（並列可能）:
   - パラメータスイープ
   - 各実験シリーズは独立に実行可能

4. **統合分析**:
   - 全実験の結果を集約
   - 最終レポート作成

### 3.2 実行環境
- **マシン**: ABCIまたはGPU環境
- **Docker**: `docker-compose-gpu.yml`を使用
- **並列化**: 実験B/C/D/Eは別々のrunディレクトリで並列実行可能

### 3.3 ログとモニタリング
- 各実験の標準出力を `run.log` に保存
- 実行時間を記録（各ステップのタイムスタンプ）
- メモリ使用量をモニタリング（必要に応じて）

---

## 4. 評価とレポート

### 4.1 評価指標
各実験について、以下を記録・分析する：

#### プロセス指標
- **実行時間**: 各ステップ（rule_pool, arms, arm_run, retrain_eval）
- **追加evidence数**: iterごとの累積
- **arm選択分布**: 各armの選択回数
- **報酬推移**: iterごとの平均報酬

#### 精度指標
- **Hits@1/3/10**: before/after、変化量
- **MRR**: before/after、変化量
- **Target score**: 対象トリプルの平均スコア（before/after）

#### リソース指標
- **メモリ使用量**: ピーク時
- **API呼び出し回数**: OpenAI（witness計算には不要だが、将来のLLM-policy用）
- **ディスク使用量**: 各ステップの出力サイズ

### 4.2 分析項目
1. **スケーラビリティ**:
   - arm数とメモリ/時間の関係
   - 反復回数と計算コストの線形性

2. **精度改善**:
   - どのパラメータ設定が最も効果的か
   - 選択戦略の影響
   - 再学習の効果

3. **arm選択の傾向**:
   - 高報酬armの特徴（cooc、ルール品質）
   - 探索と活用のバランス

4. **証拠の質**:
   - accepted evidenceのprovenance（train_removed由来 vs 新規）
   - witnessの厚み分布

### 4.3 レポート構成
最終レポート（`REC-20260116-FULL_PIPELINE_RESULTS-001.md`）は以下の構成とする：

```markdown
# 実験結果サマリー
- 実験日時
- 実行環境
- データセット

# 実験A: ベースライン
- 設定
- 実行時間
- 結果（表・グラフ）
- 考察

# （以下、実験B/C同様）

# 統合分析
- パラメータ感度分析
- 最適設定の提案
- スケーラビリティの考察

# 結論と今後の課題
```

---

## 5. 実験実行チェックリスト

### 準備（事前）
- [ ] KGEモデルの学習完了
- [ ] target_triples.txt生成
- [ ] candidate_triples.txt準備
- [ ] 設定ファイル確認（config_embeddings.json等）
- [ ] 実験ディレクトリ作成（`experiments/20260116/prep`）

### 実験A（ベースライン）
- [ ] コマンド確認
- [ ] 実行開始
- [ ] ログ監視
- [ ] 出力ファイル確認
- [ ] 結果の妥当性検証
- [ ] 実行時間記録

### 実験B/C
- [ ] パラメータ設定確認
- [ ] 各実験の実行
- [ ] 結果の集約

### 分析とレポート
- [ ] 全実験の結果を集約
- [ ] グラフ・表の作成
- [ ] 考察の記述
- [ ] レポート完成
- [ ] ドキュメント台帳更新

---

## 6. リスクと対策

### リスク1: 計算時間の超過
- **リスク**: FB15k-237全体での実験が想定以上に時間がかかる
- **対策**: 
  - 実験Aで見積もりを取得
  - 必要に応じてn_targets_per_armを削減
  - ABCIのジョブキューで長時間実行

### リスク2: メモリ不足
- **リスク**: arm数が多い場合にメモリ不足
- **対策**: 
  - max_witness_per_headで計算量を制御
  - arm数を段階的に増やす（実験B）
  - スワップやメモリプロファイリング

### リスク3: API制限
- **リスク**: OpenAI APIのrate limitに引っかかる
- **対策**: 
  - witness計算はローカルなのでAPI不要
  - 将来のLLM-policy選択時はリトライ機構を確認

### リスク4: 結果の再現性
- **リスク**: 実験結果が再現できない
- **対策**: 
  - random_seed固定（config_embeddings.json）
  - 全パラメータをログに記録
  - 実験ディレクトリを保存

---

## 7. 今後の拡張

本実験で基本的なパイプラインを確立した後、以下の拡張を検討する：

1. **順序付きarm（sequential arm）の導入**:
   - `arm_type="sequence"` のサポート
   - 段階的な証拠取得の効果検証

2. **arm動的拡張**:
   - 反復中に共起を学習してarmを追加
   - 実装: `maybe_expand_arms()` in arm_pipeline

3. **LLM-based衝突解決**:
   - 機能的矛盾の二次判定
   - 複数国籍などの例外処理

4. **他のデータセット・関係での検証**:
   - YAGO3-10
   - WN18RR
   - 他の機能的関係（place_of_birth等）

5. **オンライン学習との統合**:
   - 反復ごとに埋込モデルをincremental updateする実装

---

## 8. 参考資料

- [RULE-20260111-COMBO_BANDIT_OVERVIEW-001](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md): アルゴリズム概要
- [REC-20260114-ARM_PIPELINE-001](REC-20260114-ARM_PIPELINE-001.md): パイプライン実装
- [run_full_arm_pipeline.py](../../run_full_arm_pipeline.py): 実行スクリプト
- [build_initial_rule_pool.py](../../build_initial_rule_pool.py): ルールプール生成
- [build_initial_arms.py](../../build_initial_arms.py): arm生成
- [retrain_and_evaluate_after_arm_run.py](../../retrain_and_evaluate_after_arm_run.py): 再学習・評価

---

## 9. 実行結果（別ファイル）

実行結果（コマンド/成果物/指標/差分の整理）は、別recordに分離して管理する。

- [REC-20260118-FULL_PIPELINE_RESULTS-001](REC-20260118-FULL_PIPELINE_RESULTS-001.md)

---

## 更新履歴
- 2026-01-16: 新規作成。実験A～Eの設計、評価指標、リスク対策を整理。
- 2026-01-16: 修正。test_data_for_nationality_v3の既存データセット使用、ターゲットはtrainから抽出、実験Aで再学習を実施、witness/evidence重みの説明を追加。
- 2026-01-16: 実験Eに仮説を追加。witness重視 vs evidence重視のKGE精度への影響について、追加トリプル数・質・精度指標・効率性の観点から詳細な仮説と検証項目を整理。
- 2026-01-16: 実験B～Dのepochを100に変更し、early stopping（config_embeddings_with_stopper.json）を有効化。実験Aは10 epochのまま（動作確認用）。実験Cを3つに統合（C1: load, C2: 50iter+retrain, C3: 100iter+retrain）。
- 2026-01-16: 実験Bを削除（arm数スケーリングは不要と判断）。実験C→B、D→C、E→Dに繰り上げ。
- 2026-01-16: 実験C1も再学習を実施するように変更。C1とC2は同一設定（再現性検証用）、C3は100 iterで長期実行。
- 2026-01-16: 実験BとCを統合。Bは選択戦略×反復回数のマトリクス実験に（6実験: UCB×3種, ε-greedy×2種, Random×1種）。旧Dを新Cに繰り上げ。C1の反復回数を25に変更。
- 2026-01-16: 実験Bのε-greedyをLLM-policyに変更（より高度な選択戦略）。Randomベースラインを25/50/100 iterに拡張（合計8実験: UCB×3, LLM×2, Random×3）。

- 2026-01-18: 実行結果（9章以降）を別ファイルに分離し、結果・差分の集約を [REC-20260118-FULL_PIPELINE_RESULTS-001](REC-20260118-FULL_PIPELINE_RESULTS-001.md) に移管。
