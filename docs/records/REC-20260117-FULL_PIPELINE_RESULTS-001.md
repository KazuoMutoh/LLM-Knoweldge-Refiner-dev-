# REC-20260117-FULL_PIPELINE_RESULTS-001: Full Pipeline 実験 最終集計（20260117）

作成日: 2026-01-17
最終更新日: 2026-01-17

参照:
- 実験計画: [docs/records/REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md](REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md)
- 途中経過: [docs/records/REC-20260117-FULL_PIPELINE_INTERIM_ANALYSIS-001.md](REC-20260117-FULL_PIPELINE_INTERIM_ANALYSIS-001.md)
- 統合ランナー仕様: [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](../rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- retrain/eval仕様: [docs/rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md](../rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md)

---

## TL;DR（要約）

### 結論（この実験で言えること）

- **KGE全体指標（MRR/Hits）は、追加triple数やproxy報酬の増加だけでは改善しない**（MRR Δ は多くのrunで負、run間相関も弱い/負）。
- 一方で **target_triples のスコア差分（after-before）は per-target で大きく動く**（改善するtargetもあるが、worsened が多数派という傾向）。
- bandit（特にUCB）は **proxy報酬の最適化としては“work”**（lateで高reward armへ収束、entropy低下/偏り増）するが、**proxy最適化がMRR改善に整列していない**。
- `witness_sum` は run/arm によって accepted の代理になったりならなかったりする（相関が0〜負になるrunもあり）、**witness項が報酬のノイズ源になり得る**。

### 主要観測（原因候補）

- nationality は（近似的に）機能的関係だが、accepted evidence に **head>1tail（矛盾兆候）が少数混入**しており、指標悪化/不安定化の一因になり得る。
- arm pool は **リーク注意（bodyに国籍が入る＝既知国籍の伝播）系が 20/40** と多く、proxy報酬を稼ぎやすい一方でノイズ/矛盾の混入リスクがある。

### `?c` 周辺 triples の扱い（仕様と示唆）

- **20260117 の既存runは evidence-only** で、added evidence により導入された新規 `?c` について `?c` を含む incident triples を追加して次数を増やす挙動は行っていない（後に方針変更し、incident triples を追加する実装を追加）。
- 「`?c` の次数を上げれば target のスコアが上がるか？」については、全runをプールした相関で **弱い正相関に留まる**（例: `spearman(Δscore, max_deg_neighbor)=+0.1106`, `spearman(Δscore, n_neighbors)=+0.1569`）。
	- したがって **次数増加だけで target スコア改善を強く期待するのは難しい**（ただし周辺構造の増加やノイズ抑制と組み合わせて再検証する価値はある）。

### 次の打ち手（優先度順）

1) **報酬設計の見直し**: witnessをそのまま入れるのではなく、accepted 側（count/rate）重視へ寄せる。
2) **nationality矛盾の抑制**: `head>1tail` を採択段階で reject するか、報酬に penalty を入れる。`diagnostics.json` の conflict 定義も nationality を拾うよう更新。
3) （方針変更後の再実験）**incident triples 追加**（`?c` 次数増加）を入れた設定で、小規模に再評価（ただし単独効果は大きくない見込み）。

---

## 1. 概要

`run_full_arm_pipeline.py` による本格実験（nationalityサブセット）について、`experiments/20260117` 配下で生成された **全 `arm_run/retrain_eval/summary.json`** を集計し、最終結果をまとめる。

- runベースディレクトリ: `/app/experiments/20260117`
- データセット: `/app/experiments/test_data_for_nationality_v3`
- ターゲット関係: `/people/person/nationality`
- beforeモデル: `/app/models/20260116/fb15k237_transe_nationality`
- after_mode: `retrain`
- 監視（monitor）: 実験完了後に停止済み（PIDファイル `monitor.pid` は削除）

---

## 2. 集計対象（完了判定）

完了判定は `arm_run/retrain_eval/summary.json` の存在とし、以下を対象とする。

- A: exp_A_baseline
- Series B（selector戦略）: exp_B1_ucb_25, exp_B2_ucb_50, exp_B3_ucb_100, exp_B4_llm_25, exp_B5_llm_50, exp_B6_random_25, exp_B7_random_50, exp_B8_random_100
- Series C（重み）: exp_C1_w1.0_e1.0, exp_C2_w2.0_e1.0, exp_C3_w1.0_e2.0, exp_C4_w0.5_e1.5

---

## 3. 最終結果サマリ（KGE 指標）

注意:

- Hits/MRR は「大きいほど良い」。
- `target_score` はスコアリング方式に依存するため、単独最適化の妥当性は要検討。

| exp | n_iter | added | target_score Δ | MRR Δ | Hits@1 Δ | Hits@3 Δ | Hits@10 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_baseline | 10 | 530 | -0.2676 | -0.01481 | +0.00735 | -0.04412 | -0.04657 |
| B1_ucb_25 | 25 | 821 | -0.2565 | -0.01793 | -0.02941 | -0.01225 | +0.00980 |
| B2_ucb_50 | 50 | 892 | -0.4607 | -0.00075 | +0.00245 | -0.01716 | +0.00000 |
| B3_ucb_100 | 100 | 1011 | -0.2741 | -0.00773 | -0.01961 | -0.00490 | -0.00735 |
| B4_llm_25 | 25 | 787 | -0.2395 | +0.00716 | +0.03431 | -0.05637 | -0.00245 |
| B5_llm_50 | 50 | 1032 | -0.1199 | -0.01368 | -0.01471 | -0.03186 | -0.01225 |
| B6_random_25 | 25 | 671 | -0.2633 | -0.00395 | -0.00490 | +0.00000 | -0.01471 |
| B7_random_50 | 50 | 1108 | -0.0659 | -0.03250 | -0.02451 | -0.05882 | -0.02941 |
| B8_random_100 | 100 | 1197 | +0.2811 | -0.00299 | +0.01225 | -0.03186 | -0.02451 |
| C1_w1.0_e1.0 | 20 | 809 | -0.2090 | -0.00321 | -0.00735 | -0.00490 | -0.00735 |
| C2_w2.0_e1.0 | 20 | 801 | -0.5908 | -0.00200 | -0.00490 | -0.00490 | -0.00980 |
| C3_w1.0_e2.0 | 20 | 835 | -0.1001 | +0.00441 | +0.02206 | -0.04167 | -0.01961 |
| C4_w0.5_e1.5 | 20 | 801 | -0.2630 | -0.00679 | -0.00735 | -0.02451 | +0.00490 |

---

## 4. 所見（要点）

1) 「追加evidence数」だけでは KGE 指標改善に直結しない

2) Series B（selector戦略）では `llm_policy × 25 iter`（B4）が相対的に良い

- 同じ llm_policy でも 50 iter（B5）では改善しないため、反復増が単純に効くわけではない。

3) Series C（重み）では `evidence_weight` を上げた設定（C3）が相対的に良い

- C3（w=1.0, e=2.0）は MRR Δ が正（+0.00441）かつ Hits@1 Δ も正（+0.02206）。
- ただし Hits@3/10 は低下しており、改善は一部指標に限定。

---

## 5. accepted evidence の品質サマリ

集計スクリプト出力（debug）:

- `/app/tmp/debug/20260117_full_pipeline_evidence_stats.json`
- `/app/tmp/debug/20260117_full_pipeline_evidence_predicates_top10.md`

### 5.1 重複率（iter間の再採択）

全runで **重複率（dup rate）が 0.000** だった。

- 解釈: 同一 triple の再採択（同じaccepted evidenceが別iterで再び出る）は、この実験設定では発生していない。
- 注意: これは「追加された triple の質が高い」ことを直接意味しない（単に新規性が保たれている、という性質）。

### 5.2 nationality（機能的）矛盾の兆候

`/people/person/nationality` に限り、「同一人物 head に複数 tail が追加されている」ケースを矛盾兆候としてカウントした。

観測:

- A は 0（head>1tail=0）
- 多くのrunで head>1tail が 1〜2 程度、該当 triples は 2〜4 程度

この規模でも “少数だが矛盾が混入している” ため、KGE指標が悪化/不安定化する一因になり得る。

### 5.3 指標表（accepted evidence の統計）

| exp | iter files | accepted rows | unique triples | dup rows | dup rate | pred uniq | nationality rows | nat uniq | nat heads>1tail | nat conflicting triples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline | 10 | 530 | 530 | 0 | 0.000 | 17 | 135 | 135 | 0 | 0 |
| B1_ucb_25 | 25 | 821 | 821 | 0 | 0.000 | 17 | 175 | 175 | 2 | 4 |
| B2_ucb_50 | 50 | 892 | 892 | 0 | 0.000 | 16 | 177 | 177 | 2 | 4 |
| B3_ucb_100 | 100 | 1011 | 1011 | 0 | 0.000 | 16 | 177 | 177 | 2 | 4 |
| B4_llm_25 | 25 | 787 | 787 | 0 | 0.000 | 16 | 148 | 148 | 2 | 4 |
| B5_llm_50 | 50 | 1032 | 1032 | 0 | 0.000 | 17 | 176 | 176 | 2 | 4 |
| B6_random_25 | 25 | 671 | 671 | 0 | 0.000 | 17 | 143 | 143 | 1 | 2 |
| B7_random_50 | 50 | 1108 | 1108 | 0 | 0.000 | 17 | 185 | 185 | 2 | 4 |
| B8_random_100 | 100 | 1197 | 1197 | 0 | 0.000 | 17 | 185 | 185 | 2 | 4 |
| C1_w1.0_e1.0 | 20 | 809 | 809 | 0 | 0.000 | 16 | 175 | 175 | 2 | 4 |
| C2_w2.0_e1.0 | 20 | 801 | 801 | 0 | 0.000 | 16 | 174 | 174 | 1 | 2 |
| C3_w1.0_e2.0 | 20 | 835 | 835 | 0 | 0.000 | 16 | 172 | 172 | 1 | 2 |
| C4_w0.5_e1.5 | 20 | 801 | 801 | 0 | 0.000 | 16 | 172 | 172 | 1 | 2 |

---

### 5.4 target_triple の per-target スコア変化（Series B/C）

`iteration_metrics.json` には per-target のスコアが保存されていないため、before/after モデルで `target_triples.txt`（117件）を再スコアリングし、各target tripleのスコア差分（Δ）を算出した。

参照（再現用）:

- 集計結果（B/C 全run）: [tmp/debug/20260117_target_triple_score_change_analysis_BC_summary.md](../../tmp/debug/20260117_target_triple_score_change_analysis_BC_summary.md)
- 個別run例（B3）: [tmp/debug/20260117_target_triple_score_change_analysis_B3_ucb_100.md](../../tmp/debug/20260117_target_triple_score_change_analysis_B3_ucb_100.md)

観測（B/C 全12runに共通する傾向）:

- improved より worsened の方が多い（例: worsened 73〜94件、improved 12〜63件）。
- unknown は全runで 11件（entity/relation mapping不一致などが原因の可能性）。
- improved の平均Δは正、worsened の平均Δは負（符号は一貫）。

具体例（B3_ucb_100; added_evidence で person に incident な追加トリプルも併記）:

- improved例: `(/m/05bp8g, /people/person/nationality, /m/03_3d)`（Δ=+1.953）
	- 追加（incident to person）: `(/m/05bp8g, /film/actor/film./film/performance/film, /m/026q3s3)`
	- 追加（incident to person）: `(/m/05bp8g, /film/actor/film./film/performance/film, /m/056k77g)`
	- 追加（incident to person）: `(/m/05bp8g, /film/actor/film./film/performance/film, /m/05dfy_)`
	- 追加（incident to person）: `(/m/05bp8g, /people/person/place_of_birth, /m/07dfk)`

- worsened例: `(/m/0465_, /people/person/nationality, /m/014tss)`（Δ=-1.643）
	- 追加（incident to person）: 0件

### 5.5 仮説検証：improved は「person周辺の新規隣接?c」が多いか

仮説（ユーザ提案）:

- target = `(a, nationality, b)` に対して、added evidence により person `a` の周辺に導入される隣接エンティティ `?c` が多いほどスコアが改善しやすい。
- さらに `?c` 自体の次数（degree）が高いほど改善しやすい。

ここで `?c` は「added_evidence のうち `a` に incident な triple から得られる隣接エンティティ」と定義する:

- `(a, r, c)` または `(c, r, a)` を満たす added evidence から `c` を集めた集合。

参照（再現用）:

- 横断集計: [tmp/debug/20260117_target_triple_neighbor_hypothesis_check_BC.md](../../tmp/debug/20260117_target_triple_neighbor_hypothesis_check_BC.md)

結果（B/C 全12run; 117 targets × run）:

- **支持される部分**: `improved` 側は `?c` の数（`added_neighbor_count_person`）の中央値が大きい run が多い。
	- 例: B4 は improved median #?c=4, worsened median #?c=1 で、`spearman(Δ,#?c)=+0.353`。
- **支持されない/観測できない部分**: `?c` の「既存グラフ（before train）での次数が高い」傾向は観測できない。
	- 理由: added evidence によって接続される `?c` は、ほぼすべて before train に存在しない（median existing ?c = 0）ため、before次数は0になりやすい。
	- したがって「高次数hubに繋がるから改善する」ではなく、「新規近傍が多く追加される（周辺構造が増える）ほど改善しやすい」という形でのみ仮説が支持される。

### 5.6 仕様の確認と変更：`?c` を含む incident triples は（当時）追加されていない／追加する実装を追加

ユーザ指摘の観点（「`?c` が導入されたとき、`?c` を含む周辺 triples も追加して次数を増やしているのでは？」）について整理する。

1) **20260117 の実験成果物は evidence-only（`accepted_evidence` のみを追加）**

- 本実験（`experiments/20260117`）時点の実装では、KG更新に使う追加分は `accepted_evidence_triples.tsv`（evidence triples）のみだった。
- そのため、added evidence によって導入された新規エンティティ `?c` について、`train_removed.txt` などの候補集合から「`?c` を含む incident triples」を追加して次数を増やす挙動は行っていない。
- 補足: `rule_extractor` の k-hop サブグラフ抽出は「ルール抽出のための囲い込み」であり、KG更新方針（追加する triples の範囲）とは別。

2) **上記の方針を変更し、`?c` を含む incident triples を追加する実装を追加した**

- 受理された evidence が新規エンティティを導入した場合、その新規エンティティを含む候補 triples（例: `train_removed.txt`）を incident として追加する挙動を実装した。
- これにより、evidence 由来の `?c` の次数を（候補集合の範囲で）増やしやすくなる。
- 仕様/実装の詳細は以下を参照:
	- [docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)
	- [docs/rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md](../rules/RULE-20260117-RETRAIN_EVAL_AFTER_ARM_RUN-001.md)

注意: 20260117 の既存run（このレポートの集計対象）は方針変更前の成果物なので、当時の `updated_triples/train.txt` には「incident triples による `?c` 次数増加」は反映されていない。

### 5.7 仮説検証：`?c` の次数を上げても target_triples のスコア改善は強くは期待できない

「`(a, nationality, b)` の a に対して追加された incident triples で接続された隣接 `?c` の次数が高いほど、target triple のスコア差分（after-before）が改善するか？」を追加検証した。

参照（再現用）:

- 集計レポート: [tmp/debug/20260117_target_delta_vs_neighbor_degree.md](../../tmp/debug/20260117_target_delta_vs_neighbor_degree.md)
- per-target 点群: [tmp/debug/20260117_target_delta_vs_neighbor_degree_points.tsv](../../tmp/debug/20260117_target_delta_vs_neighbor_degree_points.tsv)

結果（全runをプール、追加近傍がある targets のみ）:

- `spearman(Δscore, max_deg_neighbor)` = **+0.1106**
- `spearman(Δscore, mean_deg_neighbor)` = **+0.0571**
- `spearman(Δscore, n_neighbors)` = **+0.1569**

解釈:

- いずれも「弱い正相関」で、`?c` の次数を上げること自体が target スコア改善を強く駆動する、というほどの効果は観測されていない。
- むしろ `?c` の次数（max/mean）よりも、`a` 周辺にどれだけ多くの近傍が追加されたか（`n_neighbors`）のほうが相対的に効いていそう、という傾向。
- ただし `n_neighbors` と `max_deg_neighbor` は相関しやすく（近傍が増えるほど max も上がりやすい）、交絡の可能性があるため「次数が直接効いている」と断定はできない。

補足（新規エンティティが degree-1 で増殖して悪化する仮説）:

- 20260117 run の成果物では「新規エンティティの degree-1 率」はほぼゼロで、run間の変動もほとんどなかった（非ゼロ 1/13）。
- したがって、この日の悪化/非改善の主因を「degree-1 新規エンティティの増殖」とみなす根拠は弱い。
- 参照: [tmp/debug/20260117_new_entity_degree1_rate_vs_metrics.md](../../tmp/debug/20260117_new_entity_degree1_rate_vs_metrics.md)

## 6. 強化学習（bandit）として work しているか：分析結果

分析に使った出力（再現用）:

- 計算スクリプト: `/app/tmp/debug/20260117_bandit_reward_analysis.py`
- 集計結果: `/app/tmp/debug/20260117_bandit_reward_analysis.md`
- 元データ: `experiments/20260117/exp_*/arm_run/iter_*/{selected_arms.json, arm_history.json, diagnostics.json}`

本節では「banditが報酬（proxy）を最適化できているか」を確認する。

### 6.1 主要結果（strategy別の平均）

（entropy_end が小さいほど、選択が少数armに集中しやすい／gini_end が大きいほど、選択の偏りが大きい）

| strategy | n_runs | mean Δreward(late-early) | mean entropy_end | mean gini_end | mean spearman(sel,avg_reward) |
|---|---:|---:|---:|---:|---:|
| llm_policy | 2 | +17.547 | 3.132 | 0.449 | +0.484 |
| random | 3 | -11.514 | 3.488 | 0.275 | -0.103 |
| ucb | 8 | +4.695 | 3.127 | 0.381 | +0.686 |

解釈:

- UCB / llm_policy は「後半の平均rewardが上がる（Δreward>0）」傾向が明確。
- random は「後半の平均rewardが下がる（Δreward<0）」傾向。
- `spearman(sel,avg_reward)` が UCB で高め（平均+0.686）で、平均rewardの高いarmがより多く選ばれがち。

### 6.2 UCB vs random（同一n_iterでの対照）

| n_iter | UCB: Δreward | UCB: entropy_end | UCB: gini_end | random: Δreward | random: entropy_end | random: gini_end |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | +6.972 (B1) | 3.203 | 0.416 | -11.164 (B6) | 3.323 | 0.294 |
| 50 | +16.710 (B2) | 2.566 | 0.663 | -14.371 (B7) | 3.545 | 0.294 |
| 100 | +16.216 (B3) | 2.152 | 0.785 | -9.007 (B8) | 3.597 | 0.236 |

解釈:

- UCB は random より entropy_end が小さく、gini_end が大きい（=より強く選択が偏る）ため、「探索→活用」が進みやすい。
- reward最適化の観点では、UCB は banditとして “workしている” と言える（少なくとも proxy reward に関して）。

### 6.3 UCB が「よく選ぶようになった arm」の具体例

分析に使った出力（再現用）:

- 計算スクリプト: `/app/tmp/debug/20260117_ucb_arm_selection_analysis.py`
- 集計結果: [tmp/debug/20260117_ucb_arm_selection_analysis.md](../../tmp/debug/20260117_ucb_arm_selection_analysis.md)
- 元データ: `experiments/20260117/exp_*/arm_run/iter_*/selected_arms.json`

方法:

- UCB runs（`selector_strategy==ucb`）について、各runの early（最初の20% iter）と late（最後の20% iter）で、選択されたarmの回数を集計。
- armは `rule_keys` と `metadata` を正規化して安定ID（`ARM-xxxxxxxxxx`）に変換し、8章の意味解釈（`semantics_ja`）と結合。

観測（全UCB runを合算）:

- UCBは late に進むほど「国籍伝播（リーク注意）」系armへの集中が強い。
	- 選択回数ベースの leak share は early **0.491** → late **0.849** に上昇。

代表例（lateで増えた上位arm; 全UCB run合算）:

| arm_id | late | early | Δ | body(labels) | semantics_ja |
|---|---:|---:|---:|---|---|
| ARM-c1e166b5b8 | 50 | 9 | +41 | 受賞候補(候補者), 国籍, 受賞者, 国籍 | 2ルール併用: 既知の国籍を受賞候補/受賞者で伝播（リーク注意） |
| ARM-bfd6f2d0c0 | 47 | 7 | +40 | 受賞候補(候補者), 国籍, 受賞候補(候補者), 国籍 | 2ルール併用: 既知の国籍を受賞候補(候補者)で伝播（リーク注意） |
| ARM-1605905cc6 | 20 | 1 | +19 | 出演作品, 作品の国, 受賞候補(対象作品), 作品の国 | 2ルール併用: 出演作品/候補作品の国から国籍を推定（弱い仮定） |

解釈:

- UCBは proxy reward を稼ぎやすいarmに収束しやすく、今回のpoolでは「国籍を含むbody（伝播）」が高rewardになりやすい構造がある。
- この収束は 7章の「proxy最適化がMRR改善に必ずしも整列しない」観測と整合する（リーク注意armが増えるほど、矛盾やノイズの混入可能性が上がり得る）。

---

## 7. 報酬設計が妥当か：分析結果

本節では「proxy報酬（witness/evidence）を最適化することが、最終目的（MRR/Hits）を代理できているか」を確認する。

### 7.1 run間相関（Spearman; Nが小さいので参考値）

`/app/tmp/debug/20260117_bandit_reward_analysis.md` より:

- spearman(MRR Δ, total_accepted): **-0.160**
- spearman(MRR Δ, total_witness): **-0.027**
- spearman(MRR Δ, nationality_rows): **-0.199**
- spearman(MRR Δ, nationality_heads>1tail): **-0.051**

解釈:

- 今回の実験では、「acceptedを増やす」「witnessを増やす」ことが、MRR改善に一貫して結びついていない（むしろ弱い負の相関）。
- つまり、banditが proxy reward を最適化しても、KGEの目的（MRR/Hits）には必ずしも整列していない。

### 7.2 witness は evidence（accepted）の proxy になっているか

各runについて、arm実行単位で `corr(witness_sum, accepted_evidence)` を計算した（`arm_history.json` の diagnostics から）。

代表例:

- B4_llm_25: pearson **+0.678**, spearman **+0.733**（witnessが accepted をそこそこ代理できている）
- B2_ucb_50: pearson **+0.040**, spearman **-0.161**（witness が accepted をほとんど代理できていない）
- B3_ucb_100: pearson **-0.076**, spearman **-0.327**（witness が accepted と逆方向になり得る）

解釈:

- UCBを長く回すと「witnessを稼げるが accepted に繋がらない arm」を引きやすくなり、witness項が報酬のノイズ（あるいはミスリード）になり得る。
- この傾向は「witness項を大きくするほど良い」という直観と逆であり、Series C の観測（C2が良くない）とも整合する。

### 7.3 機能的関係（nationality）の矛盾ペナルティについて

`diagnostics.json` の `conflict_count` は今回すべて 0 だった一方で、accepted evidence を直接集計すると nationality には head>1tail が観測されている（5.2/5.3）。

解釈:

- 現状の conflict 定義が「nationalityの機能的矛盾」を拾っていない可能性が高い。
- もし nationality を（近似的に）機能的とみなすなら、採択段階での reject か、報酬からの penalty が必要。

候補:

- `reward = w*witness_sum + e*accepted_count - c*(nat_conflict_count)`

（nat_conflict_count の定義は 5.2 と同様で良い。多国籍を許すなら、別条件で例外扱いする。）

---

## 8. Arm pool 一覧と意味的解釈（今回実際に使われた arms）

分析に使った出力（再現用）:

- 生成スクリプト: `/app/tmp/debug/20260117_arm_pool_semantics.py`
- 一覧（全40arm; union/dedup）: [tmp/debug/20260117_arm_pool_semantics.md](../../tmp/debug/20260117_arm_pool_semantics.md)
- 元データ: `experiments/20260117/exp_*/arms/initial_arms.json`

本節では、今回の実験群で **実際に使用された arm pool を全expから収集して union し、重複を除いた一覧** を示す。

観測:

- unique arms: **40**
	- Series B/C は各expで arm pool が **40**（`exp_*/arms/initial_arms.json` の件数）
	- A_baseline は arm pool が **30**（`initial_arms.txt` は 29 行で、表現形式に差がある）
- 「リーク注意（bodyに国籍が入る＝既知国籍の伝播）」: **20/40**
	- 注意: これは “チート” というより「同型関係の伝播」だが、nationalityの機能的矛盾（5.2/7.3）や、rewardのミスリード（7.2）に繋がり得る。

### 8.1 ルールの大まかな型（重複あり）

- 場所ベース（出生地/居住地/行政区→国→国籍）: 11 arm
- 作品/番組ベース（出演作品/番組の国→国籍）: 10 arm
- 受賞ベース（受賞候補/受賞者など）: 18 arm
- リーク注意（国籍を別関係から伝播）: 20 arm

※ 上の分類は `semantics_ja` 文字列に基づく粗いカテゴリで、相互に重複する（例: 受賞系かつリーク注意）。

### 8.2 代表 arm（head_coverage が高いもの）

非リーク（推定系）:

| arm_id | kind | head_coverage | pca_conf | body (Horn rule predicates) | semantics_ja |
|---|---|---:|---:|---|---|
| ARM-4d5867bfe3 | singleton | 0.349 | 0.402 | /film/actor/film./film/performance/film, /film/film/country | 出演作品の国から国籍を推定（弱い仮定） |
| ARM-1605905cc6 | pair | 0.303 | 0.435 | /film/actor/film./film/performance/film, /film/film/country, /award/award_nominee/award_nominations./award/award_nomination/nominated_for, /film/film/country | 2ルール併用（cooc=0.33）: 出演作品の国から国籍を推定（弱い仮定） / 候補作品の国から国籍を推定（弱い仮定） |
| ARM-ae1bd3e1c9 | singleton | 0.257 | 0.400 | /location/location/contains, /people/person/places_lived./people/place_lived/location | 居住地が属する国から国籍を推定 |
| ARM-e41ca7b312 | singleton | 0.219 | 0.451 | /location/location/contains, /people/person/place_of_birth | 出生地が属する国から国籍を推定 |

リーク注意（伝播系）:

| arm_id | kind | head_coverage | pca_conf | body (Horn rule predicates) | semantics_ja |
|---|---|---:|---:|---|---|
| ARM-39de37a89d | singleton | 0.340 | 0.486 | /award/award_nominee/award_nominations./award/award_nomination/award_nominee, /people/person/nationality | 既知の国籍を受賞候補(候補者)で伝播（リーク注意） |
| ARM-bfd6f2d0c0 | pair | 0.340 | 0.486 | /award/award_nominee/award_nominations./award/award_nomination/award_nominee, /people/person/nationality, /award/award_nominee/award_nominations./award/award_nomination/award_nominee, /people/person/nationality | 2ルール併用（cooc=0.85）: 既知の国籍を受賞候補(候補者)で伝播（リーク注意） / 既知の国籍を受賞候補(候補者)で伝播（リーク注意） |
| ARM-d5af6d74f7 | pair | 0.276 | 0.496 | /award/award_nominee/award_nominations./award/award_nomination/award_nominee, /people/person/nationality, /award/award_winner/awards_won./award/award_honor/award_winner, /people/person/nationality | 2ルール併用（cooc=0.55）: 既知の国籍を受賞候補(候補者)で伝播（リーク注意） / 既知の国籍を受賞者で伝播（リーク注意） |

---

## 9. 具体例：トリプル追加と witness/evidence と報酬計算

ここでは、実験出力ファイルに存在する “具体的な1イテレーション” を使って、witness/evidence/報酬を対応づける。

### 9.1 例1（w=1, e=1）：exp_B4_llm_25 の iter_1

参照:

- `/app/experiments/20260117/exp_B4_llm_25/arm_run/iter_1/arm_history.json`
- `/app/experiments/20260117/exp_B4_llm_25/arm_run/iter_1/accepted_evidence_triples.tsv`

このiterでは3本のarmが選択されており、そのうち1本（arm_id=`arm_afa0448e8fbc`）の記録は以下（arm_historyより）:

- `witness_sum = 5`
- `accepted_evidence = 8`
- `reward = 13`

このarmが “追加した具体的evidence triple” の例（accepted_evidence_triples.tsv の先頭行）:

- `(/m/018zvb, /people/person/places_lived./people/place_lived/location, /m/07z1m)`

報酬計算（witness_weight=1, evidence_weight=1）:

- `reward = 1 * 5 + 1 * 8 = 13`

### 9.2 例2（w=1, e=2）：exp_C3_w1.0_e2.0 の iter_1

参照:

- `/app/experiments/20260117/exp_C3_w1.0_e2.0/arm_run/iter_1/arm_history.json`

このiterのあるarm（例: arm_id=`arm_b9672dabe85d`）では（arm_historyより）:

- `witness_sum = 31`
- `accepted_evidence = 52`
- `reward = 135`

報酬計算（witness_weight=1, evidence_weight=2）:

- `reward = 1 * 31 + 2 * 52 = 135`

補足:

- witness は「bodyが成立した回数の合計（支持の厚み）」で、KGに直接追加されない。
- evidence は「採択された具体トリプル（accepted）」で、updated train に取り込まれ、再学習でKGEに効く。

---

## 9. 次のアクション（提案）

- （完了）bandit検証の定量集計: `/app/tmp/debug/20260117_bandit_reward_analysis.md` を今回のベースラインとして保存。
- 報酬設計の改善: witness項が accepted を代理できないケース（B2/B3）を踏まえ、witness_sum をそのまま報酬に入れるのではなく、`accepted_rate` や `accepted_count` を重視する設計へ寄せる。
- nationality矛盾の抑制: 採択段階で `head>1tail` を検知して reject（または penalty）し、`diagnostics.json` の conflict が nationality 矛盾もカウントするように定義を見直す。
- 再実験（小規模）: Series C の枠組みで、`nat_conflict` penalty の係数 c を振って（例: 0.5/1.0/2.0）短い n_iter で効果を確認。
