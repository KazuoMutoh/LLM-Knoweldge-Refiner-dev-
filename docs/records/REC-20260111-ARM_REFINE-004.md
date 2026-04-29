# REC-20260111-ARM_REFINE-004: 実データ（test_data_for_nationality_v3）での20イテレーション検証（n_targets_per_arm=10）

作成日: 2026-01-11
更新日: 2026-01-11
参照:
- [docs/records/REC-20260111-ARM_REFINE-002.md](REC-20260111-ARM_REFINE-002.md)
- [docs/records/REC-20260111-ARM_REFINE-003.md](REC-20260111-ARM_REFINE-003.md)
- [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)

---

## 0. 目的

`run_arm_refinement.py` を実データ（nationality）で **20イテレーション**実行し、
- iterごとの追加evidenceトリプル数の推移
- iterごとの選択armとreward（proxy）の推移
- （可能なら）armごとのrewardの変化

を確認し、この方法で「有用なトリプル」が取得できていそうかを考察する。

---

## 1. 検証条件

### 1.1 入力

- 初期アーム（build_initial_arms.py 出力）
  - `/app/experiments/20251216/v3_rule/iter_0/arms/initial_arms.pkl`
- ルールプール（rule_keys解決に使用）
  - `/app/experiments/20251216/v3_rule/iter_0/rules/initial_rule_pool.pkl`
- KG/候補/ターゲット
  - `dir_triples`: `/app/experiments/test_data_for_nationality_v3`
  - `target_triples`: `/app/experiments/test_data_for_nationality_v3/target_triples.txt`
  - `candidate_triples`: `/app/experiments/test_data_for_nationality_v3/train_removed.txt`

### 1.2 実行コマンド

```
python3 run_arm_refinement.py \
  --base_output_path /app/experiments/20260111/v3_arm_run_nationality_v3_20iter_nt10 \
  --initial_arms /app/experiments/20251216/v3_rule/iter_0/arms/initial_arms.pkl \
  --rule_pool_pkl /app/experiments/20251216/v3_rule/iter_0/rules/initial_rule_pool.pkl \
  --dir_triples /app/experiments/test_data_for_nationality_v3 \
  --target_triples /app/experiments/test_data_for_nationality_v3/target_triples.txt \
  --candidate_triples /app/experiments/test_data_for_nationality_v3/train_removed.txt \
  --n_iter 20 \
  --k_sel 2 \
  --n_targets_per_arm 10 \
  --max_witness_per_head 50 \
  --selector_strategy ucb
```

### 1.3 設定（重要）

- `n_iter=20`
- `k_sel=2`（各iterで2arm選択）
- `n_targets_per_arm=10`（各armで最大10ターゲットをサンプル）
- `max_witness_per_head=50`（witnessカウント上限。evidence件数の上限ではない）
- store-only判定: **target_triplesに現れるpredicate集合（今回は`/people/person/nationality`）をhypothesisとしてpendingへ**

---

## 2. 結果サマリ

出力ディレクトリ:
- `/app/experiments/20260111/v3_arm_run_nationality_v3_20iter_nt10/iter_1` ... `iter_20`

### 2.1 追加トリプル数（evidence）

- 20iter合計の追加evidence（ユニーク）: **334**
- provenance（集合照合）:
  - `train_removed` 由来: **334/334**
  - `train` 由来: **0/334**
  - どちらにも含まれない: **0/334**

補足:
- v1のlocal acquirerは候補集合（今回は `train_removed.txt`）からbody/evidenceを引くため、この結果は想定内。

### 2.2 hypothesis（store-only）

- 20iter合計の `pending_hypothesis_triples.tsv` 出力行数合計: **9**
  - ※ store-onlyなので、KGには追加されない

### 2.3 reward（proxy）分布

- 選択arm数: 40（= 20iter × 2arm/iter）
- reward統計（selected_arms.json由来）:
  - mean: 16.4 / median: 11.0 / min: 0.0 / max: 71.0

重要:
- **この20iterでは、同一armが再選択されていない（40/40がユニーク）**。
  - よって「各armのrewardが時間とともにどう変化したか（同一armの時系列）」は観測できない。
  - 代わりに、iterごとの「選択armのreward」推移と、上位reward armの一覧で傾向を見る。

---

## 3. iterごとの集計

以下は `iter_k/accepted_evidence_triples.tsv` と `iter_k/selected_arms.json` の集計。
（`selected` は `arm_id:reward` をカンマ区切りで記載）

```
iter\tadded_evidence\tpending_hypothesis\tselected(arm_id:reward)
1\t18\t1\tarm_6af063aed93b:11.0, arm_b9672dabe85d:22.0
2\t30\t0\tarm_4c82e6b2cd44:40.0, arm_bb8a7506c99e:10.0
3\t15\t2\tarm_5f74d56bd9c3:12.0, arm_55fa5f52909f:11.0
4\t3\t0\tarm_b4de5b0762b7:6.0, arm_8b7b14313003:0.0
5\t17\t1\tarm_afa0448e8fbc:18.0, arm_6e39fa38adce:10.0
6\t10\t0\tarm_c0257dd46bc1:12.0, arm_23a246527d90:3.0
7\t4\t0\tarm_18d4e0106640:9.0, arm_eb9b675aa79d:2.0
8\t14\t0\tarm_338f218d76ba:25.0, arm_588d1e8a884f:0.0
9\t9\t0\tarm_c4405768410d:3.0, arm_3f71d68e2eb1:13.0
10\t4\t0\tarm_3dd242a6aba7:6.0, arm_390349f21ba6:1.0
11\t36\t1\tarm_5afffbb78a93:29.0, arm_a9722758b97f:47.0
12\t11\t0\tarm_f4c616c68c8b:11.0, arm_5dfdeb8d2760:13.0
13\t1\t1\tarm_3e1e3eec5ba3:0.0, arm_35a07fe6cd95:5.0
14\t2\t0\tarm_17a399f69730:5.0, arm_3f4552302cc3:1.0
15\t39\t1\tarm_40ed7a8666b4:23.0, arm_3022d63057c1:71.0
16\t47\t0\tarm_da12e576d988:16.0, arm_bba8f4e23fe3:67.0
17\t14\t0\tarm_c791aa7a53e6:19.0, arm_b9724da5a9e6:5.0
18\t28\t0\tarm_278521d2c7f9:53.0, arm_6617be64cea3:15.0
19\t26\t0\tarm_35b6158f1d3f:8.0, arm_26a0f6c44f1d:42.0
20\t6\t2\tarm_4e568cb4e5b8:8.0, arm_3c992fa5accb:4.0
```

補足:
- rewardの合計が大きいiter（例: 15〜19）は、witness/accepted_evidenceが多く取れたarmが選ばれたタイミング。
- `added_evidence=0` のarmも存在（例: iter_4の `arm_8b7b14313003`）。
  - local候補上でbodyが成立しない、または成立してもevidenceが既にKGに入っている等の理由。

---

## 4. 上位rewardのarm（参考）

今回の20iterでは各armが1回ずつしか選択されていないため、上位armは「当たりを引いた」スナップショットに近い。

- top1: `arm_3022d63057c1` reward=71.0（iter_15）
- top2: `arm_bba8f4e23fe3` reward=67.0（iter_16）
- top3: `arm_278521d2c7f9` reward=53.0（iter_18）
- top4: `arm_a9722758b97f` reward=47.0（iter_11）
- top5: `arm_26a0f6c44f1d` reward=42.0（iter_19）

---

## 5. 考察（この方法で有用なトリプルが取得されたか？）

### 5.1 取得できたのは「有用そうなevidence」か

- 追加された334トリプルは全て `train_removed.txt` 由来であり、
  - 「少なくともこのデータセット作成過程では、学習用トリプルとして存在していた（がtrainから除外された）もの」
  - である。
- v1設計（evidenceのみKGへ追加、hypothesisはstore-only）に照らすと、
  - **KGの周辺事実（body/evidence）の密度を増やす**という観点では、目的に沿った“有用そうな追加”になっている。

ただし:
- 追加されたevidenceが本当に「誤りのない知識」か、または「最終ターゲット（nationality）の推論精度を改善するか」は、
  - この実験（proxy rewardのみ）からは断定できない。

### 5.2 rewardが高いarmが「安定して良い」か

- 今回は40回の選択がすべて異なるarmになった。
  - armプールが大きい状況でUCBを回すと、初期は探索が支配的になりやすく、
  - “同じarmを何度も試してrewardが収束する”挙動にならない。
- そのため「各armのreward変化（時系列）」ではなく、
  - “どのiterで大きいrewardのarmが出たか（分散が大きい）”
  - が観測された。

次に「armごとのreward推移」を取りたい場合は、例えば:
- armプールを絞る（上位N件だけ等）
- `n_iter` を増やす（探索フェーズ後に再選択が起きやすくなる）
- selectorの設定/戦略を変更し、探索率を下げる

が必要。

### 5.3 次の検証（精度への効き）

「有用性」をより強く主張するには、以下の追加評価が必要。
- iteration前後でKGEを学習し、holdoutでHits@k/MRR等を比較
- あるいは、pending hypothesis（store-only）を別ステップで確定し、正解照合する

---

## 6. 付記（再現のためのポイント）

- `--candidate_triples` を変えると取得されるevidence集合も変わる。
- `n_targets_per_arm=10` は、各armが見るターゲットの最大数なので、
  - ターゲット数が大きいと探索範囲は限定される（計算時間の上限制御にはなる）。