# REC-20260410-REWARD_DESIGN-001: 代理報酬設計の改善検討（novelty-witness + KGEスコア逆数重み）

作成日: 2026-04-10
最終更新日: 2026-04-10

---

## 1. 背景・検討の経緯

### 1.1 現行報酬式

現行の `WitnessConflictArmEvaluator`（`simple_active_refine/arm_triple_evaluator_impl.py` L93）では以下の式を使用している。

```
reward = λ_w × witness_sum + λ_e × accepted_count
```

- `witness_sum`：ルールbodyを満たす変数代入θの総数（relation priorsで重み付け）
- `accepted_count`：現KGに未存在の証拠トリプル数（= NewEvidence）
- `conflict_count = 0.0`（v1では未実装）

### 1.2 確認された問題点

| ID | 問題 | 内容 |
|---|---|---|
| P1 | witness飽和 | KGが育つほどbodyマッチが増え続ける |
| P2 | rich-get-richer | mean_rewardが高いarmがUCBで選ばれ続ける |
| P3 | 広さ不考慮 | 1ターゲットへのwitness=10と5ターゲットへのwitness=2が同報酬 |
| P4 | conflict未実装 | nationality等の機能的矛盾にペナルティなし |
| P5 | proxy-target乖離 | witnessの増減がKGEスコア改善に直結しない |

---

## 2. 報酬設計の議論と進化

### 2.1 Δwitness（差分）の検討

$$R(a) = \lambda_w \cdot \sum_t \max(0,\, w_{\text{cur}}(t) - w_{\text{prev}}(t))$$

- **利点**: P1・P2を根本解決（飽和しない）
- **問題**: 前回witnessに**他armの貢献**が含まれてしまう
  - 例: 他armがKGに `(Chicago, located_in, USA)` を追加 → 今回のΔに混入
- **具体例での確認**:
  - 前回(iter5): witness=1（Honoluluパスのみ）
  - 他arm(iter6-7): `(Obama, born_in, Chicago)` + `(Chicago, located_in, USA)` をKGに追加
  - 今回(iter8): witness=3（Honolulu + Chicago + Los_Angeles） → **Δ=+2**

### 2.2 novelty-witness の検討

$$\text{novelty-witness}(a, t) = |\{\theta : \theta \text{ は } t \text{ のbodyを満たし、} \theta \text{ が candidate トリプルを ≥1 つ使う}\}|$$

- **利点**: 今回の候補取得だけを純評価（他armの貢献を含まない）
- **問題**: KGがdenseになると候補トリプルが差し込む余地が減り、自然に低下する
- **Δwitnessとの差**（同じ上記シナリオ）:
  - Δwitness = +2（Chicagoパス＋Los_Angelesパスの両方を計上）
  - novelty-witness = +1（Los_Angelesパスのみ）
  - **乖離の原因**: Δwitnessは他armがKGに追加したトリプルによって生まれたパス(θ2=Chicago)も自armの功績として計上

| | Δwitness | novelty-witness |
|---|---|---|
| 軸 | 時間（前回との差） | 空間（今回の候補を使うか） |
| 他armの貢献 | **含む**（過大評価の原因） | 含まない |
| armの純貢献の測定精度 | 低い | 高い |

### 2.3 dense triple への過剰抑制問題

**逆witness重み付き** `1/(w_prev+1)` とnovelty-witnessを組み合わせた場合:

$$R(a) = \sum_t \frac{\text{novelty-witness}(a, t)}{w_{\text{prev}}(t) + 1}$$

二重抑制が発生する:
- denseなターゲット: novelty-witnessが低くなりやすい（分子） かつ `w_prev+1` が大きい（分母）

修正案（2効果を分離）:

$$R(a) = \underbrace{\lambda_n \cdot \text{novelty-witness}(a)}_{\text{armの純貢献}} + \underbrace{\lambda_s \cdot \sum_t \frac{\mathbf{1}[\text{novelty-witness}(a,t) \geq 1]}{w_{\text{prev}}(t)+1}}_{\text{スパース部分へのリーチボーナス}} + \underbrace{\lambda_e \cdot |\text{NewEvidence}(a)|}_{\text{実際の追加量}}$$

---

## 3. 研究仮説に基づく最終設計

### 3.1 研究仮説

> KGEのスコアが高いトリプルは周辺情報が充実しているから、同じような周辺情報を取得すれば、スコアの低いトリプルもスコアが高くなるはずである。

形式化:
```
KGEスコア(t) ≈ f(tの周辺の証拠量)
```

### 3.2 提案報酬式

$$\boxed{R(a) = \sum_{t \in T(a)} (1 - \hat{s}_{\text{prev}}(t)) \cdot \text{novelty-witness}(a,\, t) + \lambda_e \cdot |\text{NewEvidence}(a)|}$$

| 記号 | 意味 |
|---|---|
| $\hat{s}_{\text{prev}}(t)$ | 前イテレーションのKGEスコア（正規化済み0-1） |
| $(1 - \hat{s}_{\text{prev}}(t))$ | スコアが低いほど高い「改善余地」重み |
| $\text{novelty-witness}(a, t)$ | このarmが取得した候補トリプルを使うgroupingの数 |
| $\text{NewEvidence}(a)$ | 今回KGに追加される証拠トリプル（現KG未存在） |

### 3.3 この設計の利点

**研究目的との直接整合**: proxy報酬の重み付けが最終評価指標（KGEスコア）そのものに基づく。

**closed-loopの形成**:
```
KGEスコアが低いトリプルを発見
       ↓
(1-s)が高い → arm選択時の期待報酬が高い → 優先選択
       ↓
novelty-witnessが得られると報酬発生
       ↓
次イテレーションでKGEを再学習
       ↓
スコアが上昇 → (1-s)が低下 → 次回は別のトリプルへ
```

**anomaly detectionとの整合**:
- 誤ったトリプルは周辺情報取得時にノイズが増える → witness成立が困難 → 自然に低報酬

**rich-get-richer問題の解消**:
- denseなターゲット（スコアが既に高い）は `(1-s) → 0` → 報酬が逓減

**実装コスト最小**:
- $\hat{s}_{\text{prev}}(t)$ は前イテレーションの埋め込みモデルから計算済み（`embedding.score_triples()`が既存）
- 初回は $\hat{s}=0$ として全ターゲットを均等評価（自然な初期値）

### 3.4 従来設計との比較

| 設計 | 重み付けの根拠 | 問題 |
|---|---|---|
| 現行: $\lambda_w \sum_t \text{witness}(t)$ | なし | P1-P5全て残存 |
| $\lambda_w \sum_t \Delta\text{witness}(t)$ | 時間差分 | 他armの貢献を混入 |
| novelty-witness | self-arm contribution | dense tripleで低下 |
| $\sum_t \frac{\text{novelty-witness}}{w_{\text{prev}}+1}$ | スパース優先 | 二重抑制 |
| **提案: $(1-\hat{s}_{\text{prev}}) \cdot \text{novelty-witness}$** | **KGEスコア直結** | **なし（設計上の根拠が最も強い）** |

---

## 4. 実装計画

### 4.1 novelty-witness の実装

**対象ファイル**: `simple_active_refine/arm_triple_acquirer_impl.py`

現行の `count_witnesses_for_head`（`triples_editor.py`）は「全θの数」を返す。
新たに以下の関数を追加（またはフラグ追加）する:

```python
def count_novelty_witnesses_for_head(
    head_triple: Triple,
    rule: AmieRule,
    idx: TripleIndex,
    candidate_triples: set[Triple],
    max_witness: int = 50,
) -> int:
    """今回の候補トリプルを少なくとも1つ使うθの数を返す。"""
```

**判定ロジック**: DFSバックトラック中に、各θに対して `body_triples(θ) ∩ candidate_triples ≠ ∅` なら novelty-witness としてカウント。

### 4.2 KGEスコア重み付きの実装

**対象ファイル**: `simple_active_refine/arm_triple_evaluator_impl.py`

`WitnessConflictArmEvaluator.evaluate()` に以下を追加:

```python
# 前イテレーションのKGEスコアを受け取る引数を追加
def evaluate(
    self,
    acquisition: ArmAcquisitionResult,
    current_kg_triples: Iterable[Triple],
    prev_kge_scores: Dict[Triple, float] | None = None,  # 追加
) -> ArmEvaluationResult:
    ...
    # 報酬計算部分を変更
    for t, nw in novelty_witness_map.items():
        score_prev = (prev_kge_scores or {}).get(t, 0.0)
        gap = 1.0 - score_prev  # 改善余地
        reward += self.witness_weight * gap * float(nw)
```

### 4.3 パイプライン連携

**対象ファイル**: `simple_active_refine/arm_pipeline.py`

- `ArmDrivenKGRefinementPipeline` が前イテレーションの埋め込みモデルを保持しているため、`evaluate()` 呼び出し前に `embedding.score_triples(targets)` を実行してスコアを渡す。
- 初回（前イテレーションなし）は `prev_kge_scores=None` を渡し、`gap=1.0` として全ターゲット均等評価。

### 4.4 取得時の候補セット管理

novelty-witness の計算には `candidate_triples` セットが必要。

- **localモード**: `train_removed.txt` のトリプル集合をそのまま使用
- **webモード**: 今回のiter取得分のURLトリプル集合を使用（`provenance_source="web"` で判別可能）

### 4.5 実装ステップ

| ステップ | 対象ファイル | 変更内容 | 優先度 |
|---|---|---|---|
| 1 | `triples_editor.py` | `count_novelty_witnesses_for_head()` 追加 | ★★★ |
| 2 | `arm_triple_acquirer_impl.py` | novelty_witnessの計算・格納 | ★★★ |
| 3 | `arm_triple_evaluator_impl.py` | `prev_kge_scores` 引数追加・報酬式変更 | ★★★ |
| 4 | `arm_pipeline.py` | KGEスコア計算・`evaluate()`への受け渡し | ★★★ |
| 5 | テスト | `tests/test_novelty_witness_reward.py` | ★★☆ |

---

## 5. 設計ルール更新の提案

以下のrulesドキュメントを更新する必要がある:

- [`docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md`](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
  - 代理報酬式のセクションに提案式を追記
- [`docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md`](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)
  - `evaluate()` の引数変更を反映

---

## 6. 参照ドキュメント

- [`simple_active_refine/arm_triple_evaluator_impl.py`](../../simple_active_refine/arm_triple_evaluator_impl.py)（L93: 現行報酬式）
- [`simple_active_refine/arm_triple_acquirer_impl.py`](../../simple_active_refine/arm_triple_acquirer_impl.py)（L136: witness計算）
- [`simple_active_refine/triples_editor.py`](../../simple_active_refine/triples_editor.py)（count_witnesses_for_head, _backtrack_patterns）
- [`simple_active_refine/arm_pipeline.py`](../../simple_active_refine/arm_pipeline.py)（実行順序: acquire→evaluate→KG更新）
- [`docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md`](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)
- [`docs/rules/RULE-20260118-RELATION_PRIORS-001.md`](../rules/RULE-20260118-RELATION_PRIORS-001.md)
- [`docs/output/aaai_draft_20260124_ja.md`](../output/aaai_draft_20260124_ja.md)（4.5.1節: 代理報酬の現行式）
