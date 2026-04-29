# REC-20260119-TAKG_KGFIT-001: TAKG移行（KG-FIT採用）に向けたアルゴリズム整理と統合方針

作成日: 2026-01-19  
最終更新日: 2026-02-01

## 1. 背景 / 問題意識

現状のパイプラインは、静的なKGE（PyKEENのTransE等）を中心に
- ルール抽出（AMIE+）
- ルール選択（Bandit/LLM-policy）
- ルールに基づくトリプル追加
- （必要に応じて）再学習・評価

を回している。

一方で、構造（トポロジー）だけの埋込では、追加トリプルにより局所的に過制約になりやすく、スコアやランキングが悪化するケースが観測される。

これを緩和するため、**Text-attributed Knowledge Graph (TAKG)** の要素（エンティティ/リレーションにテキスト属性を持たせ、埋込の自由度を増やす）を、まずは知識グラフ埋込（KGE）の段から導入する。

本検討では、STAGEではなく **KG-FIT**（Jiang et al., 2024; arXiv:2405.16412）を採用し、**LLMをファインチューニングせず**に、外部知識（テキスト埋め込み＋LLM誘導階層）をKGEへ取り込む統合方針を整理する。

- 論文: https://arxiv.org/abs/2405.16412
- 公式実装: https://github.com/pat-jj/KG-FIT

## 2. ゴール / 非ゴール

### ゴール

- **TAKG（KG-FIT方式）を“追加導入”**し、既存の構造のみKGE（PyKEEN/TransE等）も継続利用できるようにする（併用/切替/比較を可能にする）。
- テキスト埋め込みは **OpenAI `text-embedding-3-small`** を使用し、**事前計算・キャッシュ**を前提にする（学習時にAPIは叩かない）。
- KG-FITが要求する事前計算（クラスタ/階層の構築、近傍クラスタ、親パス等）を「成果物」として明確化する。
- 既存パイプライン（arm-run / retrain-eval / relation priors 等）への統合ポイントと必要なI/F変更を明文化する。
- baseline KGE と KG-FIT-KGE の評価（Hits@k/MRR/target score等）が同一手順で比較できるよう、成果物・I/Fを揃える。

### 非ゴール（本ドキュメントではやらない）

- 実装そのもの（コード変更、実験実行、性能評価）は行わない。
- LLM（GPT-4o等）を必須化しない（**seed階層のみ**での運用も可能にする）。
- 既存KGE（PyKEENベース）を撤去/置換しない（後方互換を維持する）。

## 3. KG-FITの要点（論文に沿った整理）

KG-FITは「Fine-tune LLM with KG」ではなく「**Fine-tune KG with LLM**」を志向し、
- LLMから得た **テキスト埋め込み**（グローバル意味）
- KG構造に基づく **リンク予測目的**（ローカル意味）
- LLM誘導で得た **エンティティ階層**（グローバル構造）

を、**KGE側の学習目標に統合**する枠組み。

論文のパイプラインは大きく2段。

### 3.1 LLM-Guided Hierarchy Construction（階層構築）

1) **Entity Embedding Initialization**
- 各エンティティ $e_i$ について、LLMで短い説明文 $d_i$ を生成（または既存テキストを利用）。
- 埋め込みモデル $f$ で、エンティティ名埋め込み $\mathbf{v}^e_i$ と説明埋め込み $\mathbf{v}^d_i=f(d_i)$ を得て結合:

$$
\mathbf{v}_i=[\mathbf{v}^e_i;\mathbf{v}^d_i]
$$

2) **Seed Hierarchy Construction**
- $\{\mathbf{v}_i\}$ を対象に、agglomerative clustering（cosine距離・average linkage）で階層クラスタを構築。
- しきい値は silhouette score を最大化する $\tau_{optim}$ を選ぶ。

3) **LLM-Guided Hierarchy Refinement (LHR)**（任意）
- クラスタ分割（split）と、親子関係のbottom-up refinementにより、階層の意味整合性をLLMで改善する。

### 3.2 Global Knowledge-Guided Local Knowledge Graph Fine-Tuning（KGE微調整）

KG-FITは、任意のベースKGE（TransE/RotatE/HAKE等）に対して、
リンク予測損失に加えて「階層制約」「意味アンカー制約」を足す。

1) **Entity/Relationの初期化**
- エンティティ埋め込み $\mathbf{e}_i$ は、ランダム初期値 $\mathbf{e}'_i$ と、テキスト埋め込みを次元 $n$ に整形した $\mathbf{v}'_i$ を混合:

$$
\mathbf{e}_i=\rho\,\mathbf{e}'_i+(1-\rho)\,\mathbf{v}'_i
$$

- リレーション埋め込みはランダム初期化。

2) **制約（正則化）**
- 階層クラスタ制約 $\mathcal{L}_{hier}$（クラスタ凝集/分離/階層距離維持）
- セマンティック・アンカー $\mathcal{L}_{anc}$（学習後埋め込みがテキスト由来の意味から逸脱し過ぎないようにする）

$$
\mathcal{L}_{anc}=-\sum_{e_i\in\mathcal{E}} d(\mathbf{e}_i,\mathbf{v}'_i)
$$

3) **リンク予測目的**
- ベースKGEのスコア関数 $f_r(\cdot)$ による負例サンプリング付き目的 $\mathcal{L}_{link}$ を最適化。

4) **総目的**

$$
\mathcal{L}=\zeta_1\mathcal{L}_{hier}+\zeta_2\mathcal{L}_{anc}+\zeta_3\mathcal{L}_{link}
$$

重要ポイント:
- **テキストエンコーダ自体は学習しない**（埋め込みは固定・事前計算で再利用）。
- 既存KGEの「学習の速さ」を維持しつつ、テキスト/階層由来の外部知識をKGEへ注入する。

## 4. 本プロジェクトへの落とし込み（仕様の見直し）

### 4.1 “OpenAI埋め込み + 事前計算”の扱い

論文では `text-embedding-3-large` を利用しているが、本プロジェクトの要件として `text-embedding-3-small` を採用する。

実装上の論点:
- KG-FITは $\mathbf{v}_i=[\mathbf{v}^e_i;\mathbf{v}^d_i]$ を想定する（名前＋説明の2本立て）。
- 本リポジトリのデータでは、`entity2text.txt` / `entity2textlong.txt` があるため、以下で代替できる:
  - name相当: `entity2text.txt`
  - description相当: `entity2textlong.txt`（無ければ `entity2text.txt` を再利用）

ただし name/desc を単純に結合すると次元が $2d_t$ になり、ベースKGE（特に負例サンプリング込みのスコア計算）の計算量・メモリが概ね $O(n)$ で増える。
一方で、**slicing（先頭次元だけを使う）を固定すると重要特徴を落とす懸念**がある（特に、埋め込みが Matryoshka 的性質を前提としていない場合）。

このため、本プロジェクトでは次元整形を「固定」せず、以下の **3つの整形戦略**を仕様として用意し、実験で選べるようにする。

#### 整形戦略（提案）

1) **full（情報保持優先; 推奨）**
- $\mathbf{v}'_i := [\mathbf{v}^e_i;\mathbf{v}^d_i] \in \mathbb{R}^{2d_t}$
- KGE側の埋め込み次元は $n := 2d_t$ として扱う
- 特徴量を落とさない一方、計算量/メモリは増える

2) **slice（軽量優先; 論文のslicing互換）**
- $n$ を偶数として
  - $\mathbf{v}'_i = [\mathbf{v}^e_i[:n/2];\mathbf{v}^d_i[:n/2]] \in \mathbb{R}^n$
- 計算量/メモリを抑えられるが、情報落ちのリスクがある

3) **project（折衷; 推奨の代替）**
- まず full を作ってから、学習可能な線形射影で $n$ へ落とす

$$
  ilde{\mathbf{v}}_i=[\mathbf{v}^e_i;\mathbf{v}^d_i] \in \mathbb{R}^{2d_t} \\
\mathbf{v}'_i = W\tilde{\mathbf{v}}_i + b \in \mathbb{R}^n
$$

- 「切り捨て」ではなく情報を混ぜて圧縮するため、sliceより劣化しにくい想定
- 追加パラメータ/計算（$O(n\cdot 2d_t)$）は増えるが、LLM自体を学習しない方針は維持できる

備考:
- `text-embedding-3-small` の出力次元は上限があり（APIで縮小はできても拡大はできない）、"embedding次元を大きくする"場合は **(a) full結合で $2d_t$ を使う** か、または **(b) KGEの $n$ 自体を大きく取る**（= relation/entityの学習パラメータを増やす）ことになる。
- 計算コストが許容できるなら、まずは **full** をデフォルトにし、重い場合に slice / project を検討する。

#### FB15K-237での推奨（まずはこれで試す）

FB15K-237（#Ent ≈ 14,541）から始める前提では、**full（情報保持優先）** をデフォルトにするのが妥当。

- 例: `text-embedding-3-small` が $d_t=1536$ 次元だとすると、full では $n=2d_t=3072$。
- entity embedding 行列だけなら、float16での保持は概算で

$$
14541\times 3072\times 2\,\text{bytes} \approx 89\,\text{MB}
$$

程度で、キャッシュや推論用途では十分現実的。

注意点:
- 学習時は optimizer state（例: Adamの1st/2nd moment）や勾配でメモリが増えるため、GPUメモリが厳しい場合は
  - optimizerを軽量化（SGD系）
  - mixed precision
  - project/slice
  の優先度を上げる。
- baseline KGE との公平比較を重視する場合は、baseline側も $n$ を揃えた設定（または計算量を揃えた設定）を別途用意する。

### 4.2 LLM（説明生成/LHR）の扱い

KG-FITは
- エンティティ説明生成（LLM）
- LLM誘導階層精錬（LHR）

を含むが、コスト・安定性・再現性の観点から、本プロジェクトでは段階導入を前提にする。

- Phase 1（必須）: 既存テキスト（`entity2text*`）を説明として利用し、**seed階層のみ**で回す
- Phase 2（任意）: GPT-4o等で説明を再生成（入力/出力をキャッシュして固定）
- Phase 3（任意）: LHR（split + bottom-up refine）を導入

LLMを導入する場合は、以下を成果物として固定し、実験の再現性を担保する。
- `generated_descriptions.jsonl`（entity_id, prompt_version, model, output_text, hash）
- `lhr_actions.jsonl`（クラスタID/ノードに対するLLM提案の履歴）

### 4.3 階層の事前計算成果物

KG-FITは学習時のオーバーヘッドを下げるため、階層由来の情報を事前計算する（論文 Appendix D.1）。
本プロジェクトでも同様に、次の成果物を `dir_triples/.cache/kgfit/` に置く。

- `entity_text_embeddings__text-embedding-3-small__dim{d_t}__v1.(npy|emb)` + meta
- `hierarchy_seed.json`（クラスタ木、各leafクラスタのentity一覧）
- `hierarchy_lhr.json`（任意。LHR後の階層）
- `cluster_embeddings.npy`（各クラスタ中心 $\mathbf{c}$）
- `neighbor_clusters.json`（各クラスタの近傍クラスタID集合 $\mathcal{S}_m(C)$）
- `parent_paths.json`（各entityまたはleafクラスタの親ID列 $p_1, p_2, ...$）

### 4.4 KGE統合方針（baselineとの併用/切替）

要件: baseline KGE（既存PyKEEN）を残しつつ、KG-FITを追加導入する。

統合方針（提案）:
- `KnowledgeGraphEmbedding` に `embedding_backend` を導入
  - `pykeen`（既存）: 何も変えない
  - `kgfit`（新規）: 次を行う
    1) テキスト埋め込みのロード（事前計算）
    2) 階層成果物のロード（seed/LHR）
    3) ベースKGE（TransE/RotatE/HAKE等）を選択
    4) **リンク予測損失 + (hier + anc) 制約**で学習

PyKEEN連携の設計論点:
- KG-FITは「通常のKGE損失」に加えて、embedding同士の距離制約を追加するため、
  - PyKEENの `Regularizer` だけでは表現しづらい
  - **カスタムTrainingLoop**または **Model側での追加損失**が必要になる可能性が高い

このため、実装段階では次の2案を比較する（本ドキュメントでは決定しない）。

- 案A: PyKEEN上で `Model`/`TrainingLoop` を拡張してKG-FIT損失を組み込む（評価/保存互換を最大化）
- 案B: 学習は独自ループ（PyTorch）で行い、評価のみPyKEENのEvaluator互換I/Fに合わせる（実装容易性を優先）

### 4.5 既存周辺コードへの影響

- relation priors（`relation_priors_compute.py`）は entity embedding 行列が必要。
  - KG-FIT backendでも、学習後の entity embedding を **全件取り出し可能**にする必要がある。
- `score_triples()` は `model.score_hrt()` 相当のI/Fを保つか、backend分岐で互換を提供する。

## 5. 実装TODO（仕様ベース）

実装は別チケットだが、仕様として必要な作業を列挙する。

1) 事前計算コマンド
- entity text embeddings 生成（OpenAI）
- seed hierarchy 構築（clustering + silhouette）
- （任意）LHR（LLM actionsのキャッシュ）
- 学習用の precompute（cluster centers / neighbors / parent paths）

2) KG-FIT学習
- ベースKGEの選択（TransE/RotatE/HAKE等）
- $\mathcal{L}_{hier}, \mathcal{L}_{anc}$ の実装
- 負例サンプリングと $\mathcal{L}_{link}$ の実装（既存PyKEENに寄せるか独自か）

3) 成果物と互換
- baselineと同一の保存/ロード導線
- arm-run / retrain-eval から `embedding_backend` を指定可能にする

## 6. 更新履歴

---

## 結論（観測→含意→次アクション）

### 観測

- 構造のみKGEでは追加トリプルにより局所的に過制約になりやすく、指標悪化が観測されるため、TAKG要素をKGE段から導入する方針を明確化した。
- STAGEではなくKG-FITを採用し、テキスト埋め込み（固定・事前計算）と（任意で）LLM誘導階層を、KGEの目的関数へ統合する設計を整理した。
- 運用上の最重要点として「学習時にAPIを叩かない」「成果物（.cache/kgfit/）を固定して再現性を担保する」「baseline KGEと併用/切替できるI/Fを揃える」を明文化した。

### 含意

- KG-FIT導入の成功可否は、事前計算成果物の設計（embedding/seed階層/neighbor/parent path）と、学習I/Fの後方互換（score_triples/evaluate/保存）に強く依存する。
- “full/slice/project” の次元整形戦略は、計算資源と性能のトレードオフを直接左右するため、デフォルト（full）とフォールバック（slice/project）を明確に分けて運用すべきである。

### 次アクション

- `.cache/kgfit/` 成果物の生成導線（テキスト埋め込み事前計算→seed階層→neighbor clusters）を安定化し、固定成果物で学習が再現できることを確認する。
- KG-FITバックエンドの統合方式（PyKEEN拡張 vs 独自学習ループ）を決め、arm-run / retrain-eval から `embedding_backend=kgfit` を指定可能にする。
- baseline（pykeen/TransE 等）と同一プロトコルで比較できるよう、評価と保存/ロードのI/Fを揃えた上で、full/slice/project の感度を実験で検証する。

- 2026-01-19: 新規作成（TAKGの埋め込み方式をSTAGEからKG-FITへ切替）
