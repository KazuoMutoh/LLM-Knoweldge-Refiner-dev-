# REC-20260117-ARM_SELECTOR-001: LLMに「armの意味と取得トリプル」を見せて次armを選ばせる設計

作成日: 2026-01-17
最終更新日: 2026-01-17

## 背景 / 課題

現状のLLMベースarm選択（`LLMPolicyArmSelector`）は、armごとの集計統計（平均報酬・直近性能等）を中心に選択していた。
しかし、armが本当に **target_triple を説明するのに有益**だったか（= targetに対して十分なwitnessを持ち、関連するevidenceを提示したか）は、集計統計だけでは判断が難しい。

## 目的

次のarm選択において、LLMが以下を参照して判断できるようにする。

- armの意味（arm_type / rule_keys）
- armが実際に取得した evidence triples（既存KG含む）
- 新規に追加された evidence triples（accepted / KGに無かったもの）
- target_triple との対応（witness / coverage / evidence relevance）

これにより、exploration/exploitation だけでなく **「説明として有益か」** を反映した選択方針（`policy_text`）の更新を促す。

加えて、**意味的な整合性（semantic alignment）** を選択の判断材料に含める。
例: targetが nationality のとき、友人関係よりも地理/出生/居住などのlocation系relationの方が説明に有益である可能性が高い。

## 観測データ（実装で収集するもの）

各iterationで、armごとに次を収集する。

- `targets_total`: そのarmが評価した target_triples 数
- `targets_with_witness`: witness>0 の target_triples 数
- `target_coverage`: `targets_with_witness / targets_total`
- `witness_sum`: targetごとのwitness合計
- `mean_witness_per_target`: `witness_sum / targets_total`
- `evidence_total`: 取得した evidence triples 総数（既存KG含む）
- `evidence_new`: KGに無かった新規evidence数（=追加候補）
- `evidence_existing`: 既存KGにあったevidence数
- `evidence_new_overlap_rate_with_targets`: 新規evidenceが targetのエンティティ（subject/object）と共有する割合

加えて、LLMに「中身」を見せるために、以下も保存する。

- `evidence_triples`: 取得したevidenceの全リスト
- `witness_by_target`: 各targetに対するwitness数
- `added_triples`: 新規に追加されたevidence（= `evidence_new`）

また、意味的判断の手掛かりとして次を利用できるようにする。

- `relation2text.txt` 由来の **relation description**（利用可能な場合）
- 各armの rule_keys から導出した **body_predicates/head_predicates** とその説明（利用可能な場合）

## 選択アルゴリズム（LLMポリシーの意図）

LLMが `policy_text` を更新する際に、次のような判断規準を自然言語で“再利用可能”な形にする。

### 1) 「説明度（explanatory usefulness）」を重視

- **target_coverage が高いarm**: 多くのtargetに対してwitnessが立つ（説明可能なtargetが多い）
- **mean_witness_per_target が高いarm**: 1 targetあたりの説明力が強い

### 2) evidenceの「関連性」を重視

- **evidence_new_overlap_rate_with_targets が高いarm**: 新規evidenceが targetの周辺を説明している可能性が高い
- 逆に、overlapが低いのに evidence_new が多いarmは、ノイズ/オフトピックの可能性がある

### 3) exploration/exploitation の両立

- UNTRIED/少試行armを一定数混ぜる（探索枠）
- それ以外は「説明度」＋「安定した報酬（mean/std/recent）」を満たすarmを優先（活用枠）

### 4) 次arm選択の具体例（policy_textに書くべき形）

- 例: 「k個中1つはUNTRIED、残りは `target_coverage>=0.3` かつ `mean_witness_per_target` 上位を優先。`evidence_new` が多いが `overlap<0.2` のarmは避ける。」

※ 実際の閾値はデータセット/iterationに依存するため、LLMは提示された統計とサンプルを見て調整する。

### 5) 意味的整合性（semantic alignment）を組み込む

定量指標（witness/coverage）だけでは拾えない「ドメイン知識」を、LLMの判断に明示的に取り込む。

- target predicate(s) の説明（`relation2text.txt`）を提示し、"この関係を説明するのに自然な証拠"を考えさせる
- 各armの body_predicates（および説明）を提示し、targetと同じ語彙/領域（地理・職業・家族・教育など）に寄っているarmを優先させる
- 例（nationality）:
	- 優先しやすい: place_of_birth / location / containedby / country / region など
	- 一般に弱い可能性: friendship / romantic_relationship のみで nationality を推定するようなarm

ただし、データ上有効な例外（友人関係が強く効く特定領域など）もあり得るため、完全なハードルールではなく「探索枠」で検証できる形にする。

## 実装方針（どこを変えるか）

1. `ArmDrivenKGRefinementPipeline` で、armごとの `targets / witness / evidence` を集計し `ArmEvaluationRecord` に保存する。
2. `LLMPolicyArmSelector` のプロンプトに、集計値だけでなく「直近のtarget例」「追加evidence例」「取得evidence例」を含める。
3. 可能なら `relation2text.txt` を読み込み、target predicate と armのbody predicatesの説明を提示して、意味的整合性を評価できるようにする。
3. `selected_arms.json` にも診断情報を出して、実験ログから人間が妥当性確認できるようにする。

## 変更箇所（コード参照）

- `ArmEvaluationRecord` に `evidence_triples` と `witness_by_target` を追加
- pipelineで説明度/関連度のdiagnosticsを計算して保存
- LLMプロンプトに「armの意味＋取得/追加evidence＋target説明度」を含める

## 期待される効果

- LLMが「平均報酬が高いarm」だけでなく「targetを説明しているarm」を選びやすくなる
- `policy_text` が“次回も使える具体ルール”として蓄積され、選択の再現性が上がる

## 既知の限界 / 今後

- 現状のrelevance指標は「targetエンティティとの共有」中心であり、関係タイプの整合性や多段推論の妥当性までは見ていない。
- 将来的には、evidenceを targetごとに束ねて「どのtargetをどう説明したか」の要約（top witness paths等）を生成して提示するとさらに良い。
