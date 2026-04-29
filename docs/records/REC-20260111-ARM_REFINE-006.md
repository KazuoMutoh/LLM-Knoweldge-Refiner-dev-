# REC-20260111-ARM_REFINE-006: arm=10・iter=100・n_targets_per_arm=5 再実験（nationality v3）

作成日: 2026-01-11  
最終更新日: 2026-01-11

## 目的
[REC-20260111-ARM_REFINE-004.md](REC-20260111-ARM_REFINE-004.md) の20iter実験ではarmが再選択されず（40/40ユニーク）、**各armの報酬推移**が観測できなかった。  
そこでarm数を10に縮小（singletonのみ）し、iter=100で再実験して「iterごとの追加トリプル数」「armごとの報酬推移」を集計する。

## 実験設定
- データ: `test_data_for_nationality_v3`
  - triples dir: `../../experiments/test_data_for_nationality_v3`
  - target: `../../experiments/test_data_for_nationality_v3/target_triples.txt`
  - candidate: `../../experiments/test_data_for_nationality_v3/train_removed.txt`
- 初期armプール（arm=10）
  - 生成済み: `../../experiments/20260111/v3_initial_arms_nationality_v3_ruleTop10_kpairs0/initial_arms.pkl`
  - 生成条件（参考）: rule_pool を top-10 (sort_by=`pca_conf`) に事前フィルタ + `k_pairs=0`
- 実行パラメータ
  - `n_iter=100`
  - `k_sel=2`
  - `n_targets_per_arm=5`
  - selector: `ucb` (c=1.0)
  - `max_witness_per_head=50`

## 実行コマンド
```bash
python3 run_arm_refinement.py \
  --base_output_path /app/experiments/20260111/v3_arm_run_nationality_v3_100iter_arm10_nt5 \
  --initial_arms /app/experiments/20260111/v3_initial_arms_nationality_v3_ruleTop10_kpairs0/initial_arms.pkl \
  --rule_pool_pkl /app/experiments/20251216/v3_rule/iter_0/rules/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_nationality_v3 \
  --target_triples /app/experiments/test_data_for_nationality_v3/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_nationality_v3/train_removed.txt \
  --n_iter 100 \
  --k_sel 2 \
  --n_targets_per_arm 5 \
  --max_witness_per_head 50 \
  --selector_strategy ucb
```

## 出力
- run dir: `../../experiments/20260111/v3_arm_run_nationality_v3_100iter_arm10_nt5/`
- 集計（本REC用）
  - `../../experiments/20260111/v3_arm_run_nationality_v3_100iter_arm10_nt5/summary/per_iter.csv`
  - `../../experiments/20260111/v3_arm_run_nationality_v3_100iter_arm10_nt5/summary/per_arm.csv`
  - `../../experiments/20260111/v3_arm_run_nationality_v3_100iter_arm10_nt5/summary/timeseries_*.csv`

集計スクリプト: `../../tmp/debug/summarize_arm_run.py`

## 結果（iterごとの追加evidence数）
- accepted evidence（tsv行数ベース）
  - 合計: 152
  - 平均: 1.52 / iter
  - 中央値: 0.5
  - 最大: 10（iter=4）
  - 追加0のiter: 50/100
- witness_total
  - 平均: 2.93 / iter
  - 中央値: 3.0
  - 最大: 7.0（例: iter=14, 55 など）
- pending hypothesis
  - 合計: 4（各iterで最大1）
  - 発生iter: 23 / 69 / 84 / 86

「追加数が大きいiter（上位）」例:
- iter=4: +10
- iter=14: +9
- iter=26: +7

## 結果（provenance）
accepted evidence 152本の由来（`train_removed.txt` / `train.txt` 照合）:
- `train_removed.txt`: 152
- `train.txt`: 0
- その他: 0

つまり、今回も **evidence-onlyの追加は train_removed 由来のみ** だった。

## 結果（armの選択分布と報酬）
100iter × k_sel=2 なので pull総数は200。

| arm_id | pulls | mean_reward | reward_sum | nonzero_rate |
|:--|--:|--:|--:|--:|
| arm_afa0448e8fbc | 88 | 2.523 | 222.0 | 0.89 |
| arm_eb9b675aa79d | 40 | 2.125 | 85.0 | 0.75 |
| arm_c0257dd46bc1 | 24 | 2.250 | 54.0 | 0.83 |
| arm_3f71d68e2eb1 | 23 | 2.043 | 47.0 | 0.57 |
| arm_588d1e8a884f | 15 | 1.867 | 28.0 | 0.47 |
| arm_3dd242a6aba7 | 6 | 1.500 | 9.0 | 0.50 |

残り4armはpull=1で、rewardは0。

## 結果（arm報酬の推移：前半 vs 後半）
「iter 1–50」と「iter 51–100」に分け、**そのarmが選ばれたpullにおける平均報酬**を比較:

| arm_id | pulls(前半) | mean_reward(前半) | pulls(後半) | mean_reward(後半) |
|:--|--:|--:|--:|--:|
| arm_afa0448e8fbc | 38 | 3.053 | 50 | 2.120 |
| arm_eb9b675aa79d | 23 | 2.435 | 17 | 1.706 |
| arm_c0257dd46bc1 | 2 | 1.000 | 22 | 2.364 |
| arm_3f71d68e2eb1 | 20 | 2.350 | 3 | 0.000 |

## 意味的分析（上位armの「取得パターン」と実際に追加されたトリプル）
この節では、よく選択されたarmが **意味的にどのような「証拠（evidence）トリプル」を取得する腕**だったか、また実際にどんな関係（predicate）がどれくらい追加されたかをまとめる。

参照した集計:
- `../../experiments/20260111/v3_arm_run_nationality_v3_100iter_arm10_nt5/summary/semantic_arm_analysis.json`
- 生成スクリプト: `../../tmp/debug/semantic_arm_analysis.py`

### 1) arm_afa0448e8fbc（pull=88）: 居住地→国（location→country）を埋める腕
- ルールbody: `/people/person/places_lived.../location` と `/base/biblioness/bibs_location/country`
- 直感: 「人が住んだ場所の属する国」から nationality を支持する。
- 実際に追加されたevidence（計60本）
  - `/people/person/places_lived.../location`: 37
  - `/base/biblioness/bibs_location/country`: 23
- 代表例（evidence）
  - `(/m/08c7cz, /people/person/places_lived.../location, /m/04jpl)`
  - `(/m/04jpl, /base/biblioness/bibs_location/country, /m/07ssc)`

**精錬に有効か**: 人→居住地、居住地→国の連結は nationality 推論に直結する「文脈辺」で、後段でembedding再学習や別ルール適用を行う場合にも効きやすい（人物と国の間の説明可能な経路が増える）。

**なぜよく選択されたか**: witness/追加evidenceが安定して発生し、UCBが高平均報酬の腕として早期に収束したため（前半で平均報酬が高く、後半は枯渇傾向で低下）。

### 2) arm_eb9b675aa79d（pull=40）: ノミネート作品→番組の国（作品/番組属性）で支える腕
- ルールbody: `/award/.../nominated_for` と `/tv/tv_program/country_of_origin`
- 直感: 「人物がノミネートされた番組（作品）の国」から nationality を支持する。
- 実際に追加されたevidence（計28本）
  - `/tv/tv_program/country_of_origin`: 14
  - `/award/.../nominated_for`: 14

**精錬に有効か**: 直接の出生/居住より弱いが、人物→作品→国の“活動圏”経路を増やし、特定ドメイン（俳優/テレビ）では nationality の代理シグナルになり得る。

**なぜよく選択されたか**: このデータでは候補にこの種の辺が一定量存在し（train_removed由来）、witnessと追加が継続したため。

### 3) arm_c0257dd46bc1（pull=24）: 出生地→国（birthplace→country）を埋める腕
- ルールbody: `/people/person/place_of_birth` と `/base/biblioness/bibs_location/country`
- 直感: 「出生地の属する国」から nationality を支持する。
- 実際に追加されたevidence（計24本）
  - `/people/person/place_of_birth`: 16
  - `/base/biblioness/bibs_location/country`: 8

**精錬に有効か**: nationality に対して最も解釈可能で強い因果的（少なくとも説明的）な手掛かり。出自関連の欠損を埋めることで、他のルール適用や将来の再学習で恩恵が大きい。

### 4) arm_3f71d68e2eb1 / arm_588d1e8a884f / arm_3dd242a6aba7: ソーシャル/交際ネットワークで nationality を“伝播”させる腕
- ルールbody（例）: `/base/popstra/.../(friendship|dated)/participant` と `/people/person/nationality`
- 直感: 「関係（友人/交際）にある相手の nationality から本人の nationality を支持する」。
- 実際に追加されたevidence
  - arm_3f71d68e2eb1: 21本（friendship 12 + nationality 9）
  - arm_588d1e8a884f: 15本（dated 8 + nationality 7）
  - arm_3dd242a6aba7: 4本（friendship 2 + nationality 2）

**精錬に有効か（注意点込み）**:
- 人間関係による属性伝播は“それっぽい”が、誤りも増えやすい（友人/恋人と国籍が一致する保証はない）。
- さらに今回の実装では、bodyに `/people/person/nationality` を含む場合、その nationality 自体が **evidenceとしてKGに追加**され得る（今回も追加evidenceの一部が nationality になっている）。
  - これは「target relation は store-only」という狙いに対して、**実質的に target 辺の再導入**になりうるため、解釈・評価上の注意が必要。

### 補足: ほとんど機能しなかったarm
`/tv/.../regular_cast` などの腕は1回は選択されたが、追加evidenceが0だった（候補不足、または既にKG内にあり新規性が無い、など）。UCBはこの情報で探索を打ち切った。

## 考察
- arm数を10に絞ったことで、**同一armが何度も再選択され、報酬推移が観測可能**になった（目的達成）。
- 一方でUCBは強く集中し、pullの大半が上位1〜2armに割り当てられた（例: arm_afa… が44%）。探索が薄いarmが複数残り、reward=0のarmは早期に見切られた。
- 上位arm（arm_afa…, arm_eb9…）は後半で平均報酬が低下しており、
  - 追加可能なevidenceの枯渇（train_removed候補の使い尽くし）
  - KGへのevidence追加により、同じターゲット集合で得られる新規witness/証拠が減る
  のいずれか（または両方）が起きている可能性がある。
- pending hypothesis は4件と少数で、今回の条件では「仮説（target relation）の生成」は限定的だった。

## 次の確認（必要なら）
- rewardが高いのに accepted_evidence が0になるケースが多いので、
  - `accepted_evidence_triples.tsv` と `witness_total` の関係（評価関数の寄与割合）
  - 追加候補の重複/フィルタで落ちている割合
  を追加で点検すると、改善余地（受理条件・候補生成）が切り分けやすい。
