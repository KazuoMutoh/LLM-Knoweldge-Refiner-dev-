# Simple Active Refinemen for Knowledge Graph

## 概要
知識グラフに不足するエンティティやトリプルを，知識グラフの外部リソースから取得することによって，知識グラフの品質を向上させます．

なお，知識グラフの品質には様々な指標がありますが，知識グラフの埋め込みモデルの精度を知識グラフの品質としてとらえ，それを向上させます．

## 新機能: TextAttributedKnowledgeGraph

`TextAttributedKnowledgeGraph`クラスは、テキスト情報を持つ知識グラフの管理と検索機能を提供します。

### 主な機能

1. **永続化とキャッシュ**: 一度初期化したら、次回はキャッシュから高速読み込み
2. **エンティティ検索**: ベクトル、キーワード、ハイブリッド検索に対応
3. **類似エンティティ検索**: Entityインスタンスに基づく類似エンティティの検索
4. **エンティティ/トリプル追加**: 新しいエンティティやトリプルの追加と自動インデックス更新
5. **型付きデータアクセス**: Entity/Tripleクラスのインスタンスとして取得

詳細な使用方法は [KNOWLEDGE_RETRIEVER_USAGE.md](./KNOWLEDGE_RETRIEVER_USAGE.md) を参照してください。

### クイックスタート

```python
from simple_active_refine.knoweldge_retriever import TextAttributedKnoweldgeGraph

# 初期化（キャッシュがあれば自動的に読み込み）
kg = TextAttributedKnoweldgeGraph(dir_triples="data/FB15k-237")

# エンティティ検索
results = kg.search_entities_by_text("American film director", method="hybrid", top_k=10)

# 全エンティティ・トリプルの取得
all_entities = kg.get_all_entities()
all_triples = kg.get_all_triples()
```

テストスクリプト:
```bash
python test_knowledge_retriever.py
```

## アルゴリズム
下記の手順で知識グラフの精度を向上させます．

### 入力
- 対象リレーション $r_t$
- 知識グラフ[^1]
- 外部リソース

[^1]:トリプルで表現されていることが前提．Propery Graphではない．

### 出力
- トリプルが追加された知識グラフ
- 上記の知識グラフのHits@kなどの評価指標

### 処理手順
1～6の処理を，決められたバジェット（＝追加するエンティティ数）がなくなるまで繰り返す．

1. 知識グラフより知識グラフ埋込モデルを計算し，各トリプルのスコアや，Hit@kなどの各種指標を計算する．
2. 対象リレーション$r_t$を持つトリプルの内，知識グラフ埋込モデルのスコアが高いトリプルを抽出し，そのk-hop囲い込みグラフを抽出する．
3. 抽出されたk-hop囲い込みグラフ[ref](#)で頻出するグラフパターンを抽出する．例えば，AMIE+[ref](#)などを用いて，Headが対象リレーション$r_t$となる，Horn Ruleを抽出する．抽出したHorn Ruleの内，ConfidenceやHead Coverageの大きいルールを"トリプル追加ルール"$R_{add}$とする．なお，ConfidenceやHead Coverageの他に，LLMなどに意味的に重要なルールを選択させても良い．
4. 対象リレーション$r_t$を持つトリプルの内，知識グラフ埋込モデルのスコアが低いトリプルを抽出し，トリプル追加ルール"R_{add}"のBodyと比較し，足りないエンティティやリレーションを特定する．
5. 足りないエンティティやリレーションを外部リソースから取得する．
6. 取得したエンティティやリレーションをトリプルとして知識グラフに追加する．


### 実装
1. 検証用ディレクトリを作成する．ディレクトリ名は日付＋時間で作成する.
2. 検証用ディレクトリの中にiter-1というディレクトリを作り，オリジナルのデータセットから作ったテストデータセットをdataというディレクトリに保存する．
下記の操作をk=1, 2, 3で，繰り返す．
3. テストデータセットから知識グラフ埋込モデルを学習し，ディレクトリiter-{k}の下にmodelというディレクトリを作成し，そこに保存する．
4. modelに基づきルールを抽出し，ディレクトリiter-{k}の下のrulesというディレクトリの保存する．
5. rulesに基づき更新するデータを抽出しデータセットに追加する．追加したデータセットはiter-{k}の下にupdated_dataというディレクトリを作成し，そこに保存する．
6. iter-{k+1}というディレクトリを検証用ディレクトリに作成しか，updated_dataをdataという名称に変えて保存する．


各iter-{k}のディレクトリ構成は下記のようになる
iter-{k}
- data
- model
- rules
- updated_data
- evaluations

### 使い方

#### 一括処理

下記のコマンドを実行してください．

```
python refine_knoweldge_graph.py {dir_triples}  {target_relation} {external_resource_type} {...other options} 
```

ここで，dir_triplesは知識グラフのエンティティやトリプル（train.txt，test.txt，valid.txt ...など）に関するデータが格納されているディレクトリへのパスです．

target_raltionは，対象リレーション$r_t$です．dir_triplesで指定される知識グラフに含まれるリレーションである必要があります．

external_resource_typeは外部リソースのタイプです．外部リソースには，triplesとinternetの二つのタイプがあります．triplesは検証用であり，知識グラフからあらかじめ除去したトリプルです．triplesが指定された時は，あらかじめ除去したトリプルが格納されているディレクトリを追加で指定します．


#### 知識グラフ埋込モデル学習

下記のコマンドを実行してください．

```
python learn_knoweldge_graph_embedding.py {dir_triples} {dir_embeding_model} {...other options} 
```

ここで，dir_embedding_modelは学習済の知識グラフ埋込モデルを格納するディレクトリです．

#### トリプル追加ルールを抽出

下記のコマンドを実行してください．

```
python extract_rules.py {dir_embeding_model} {target_relation} {score_threshold} {dir_rules}
```

{score_threshold}で，どのスコア以上のトリプルからルールを抽出するか指定します．
{dir_rules}は，抽出されたトリプル追加ルールを格納するディレクトリです．

#### 知識グラフへのトリプル追加

下記のコマンドを実行してください．

```
python add_triples_based_on_rules {dir_triples} {dir_rules} {dir_new_triples} {external resources} {...other options}
```

{dir_new_triples}は，トリプルを追加した知識グラフを格納するディレクトリです．


#### テスト用データセットの作成

下記のコマンドを実行してください．

```
python make_test_triples {dir_triples} {dir_test_triples} {target_entities} {target_relation} {target_preference} {remove_preference} {drop_ratio} {base_triples} {... other options}

ここで，dir_test_triplesはテスト用データセットを格納するディレクトリ，target_entitiesは対象エンティティ，target_relationは対象リレーションです．target_preferenceはheadもくしはtail，remove_preferenceはheadもしくはtail，bothを選択します．drop_ratioは0～1までのfloatで指定します．base_triplesは，テストデータのベースとなるデータセットであり，train，test，validのいずれかを選択します．規定値はtrainです．

テスト用データセットの作成では，base_triplesに対して，下記のような処理を行います．

1. target_preferenceがheadの時は，対象リレーション$r_t$を持ち，かつ，headが対象エンティティ$e_t$，target_preferenceがtailの時は，tailが対象エンティティ$e_t$のトリプルを，対象トリプルとして抽出します．
2. 対象リレーション以外のリレーションで対象エンティティ$e_t$に関連付くエンティティを抽出し，drop_ratioの割合で選択します．なお，remove_perferenceがheadに設定されている場合は対象エンティティから見てheadの関係で関連付いているエンティティを抽出します．また，remove_perferenceがtailに設定されている場合は対象エンティティから見てheadの関係で関連付いているエンティティを抽出します．
3. 選択したエンティティを含むトリプルを削除トリプルとします．
4. 削除トリプルをbase_triplesから取り除きます．
5. 削除トリプルが取り除かれたbase_triples，削除トリプル，base_triplesで指定されなかったデータセットをdir_test_triplesに保存します．


### 実装の制約
- 基本的にはpythonで実装します．ただし，AMIEはjavaで記述されているため，それをラップするpythonの個コードを記述してください．
- 知識グラフの埋め込みモデルの計算にはpykeenを使います．ただし，将来的にはText Attributed Knowledge Graphにも対象を拡大するためにKG-FIT[ref](#)などもにも対応できる実装にしています．
- CPU計算するか，GPUで計算するかを選択できます．
- Windows，LinuxのどちらのOSでも作動するようにしてください．

## 補足
### k-hop囲い込みグラフ
{ここを埋めて}

### Horn Rule
{ここを埋めて}

### AMIE+
{ここを埋めて}