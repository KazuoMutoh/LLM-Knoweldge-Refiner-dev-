# Agents Documentation

このドキュメントでは、LLM Knowledge Refinerプロジェクトにおける各エージェント（モジュール）の役割と責務を説明します。

---

## 概要

本プロジェクトは、知識グラフの品質向上を目的とした研究プロジェクトです。知識グラフ埋込モデルとLLM（Large Language Model）を組み合わせ、外部情報源から知識を取得して知識グラフを段階的に改善します。

### アーキテクチャの特徴

- **反復的改善**: 埋込モデルの学習 → ルール抽出/生成 → 情報取得 → トリプル追加のサイクルを繰り返す
- **LLM活用**: ルール生成・更新、外部情報の取得・検証にLLMを活用
- **品質重視**: スコアベースのフィルタリングにより、確実性の高いトリプルのみを追加

---

## エージェント一覧

### 1. KnowledgeGraphEmbedding (`simple_active_refine/embedding.py`)

**役割**: 知識グラフ埋込モデルの学習とトリプルスコアの計算

**主な機能**:
- PyKEENを用いた知識グラフ埋込モデル（TransE等）の学習
- トリプルの尤もらしさスコアの計算
- スコア分布に基づくトリプルのフィルタリング（パーセンタイル指定）
- モデルの永続化と読み込み

**入力**:
- トリプルデータ（train.txt, valid.txt, test.txt）
- モデルパラメータ（モデルタイプ、エポック数等）

**出力**:
- 学習済み埋込モデル
- トリプルのスコア（正規化済み）

**主要メソッド**:
- `train_model()`: 埋込モデルの学習
- `score_triples()`: トリプルのスコア計算
- `filter_triples_by_score()`: スコアに基づくフィルタリング
- `evaluate()`: Hit@k等の評価指標の計算

**備考**:
- スコアは0-1の範囲に正規化される
- GPU対応（利用可能な場合は自動的にGPUを使用）

---

### 2. RuleExtractor (`simple_active_refine/rule_extractor.py`)

**役割**: 高スコアトリプルからAMIE+を用いてHornルールを抽出

**主な機能**:
- 高スコアトリプルの選択（パーセンタイル指定）
- k-hop囲い込みグラフの抽出
- AMIE+による頻出パターン（Hornルール）の抽出
- ルールの品質フィルタリング（head coverage, PCA confidence等）

**入力**:
- 知識グラフ埋込モデル
- 対象リレーション
- 閾値パラメータ（lower_percentile, min_head_coverage等）

**出力**:
- AmieRulesオブジェクト（抽出されたHornルール集合）

**主要関数**:
- `extract_rules_from_high_score_triples()`: メイン処理
- `get_related_triples()`: k-hop囲い込みグラフの抽出

**備考**:
- UUID接頭辞を用いて、複数のサブグラフをマージ時の衝突を回避
- 並列処理によりサブグラフ抽出を高速化

---

### 3. BaseRuleGenerator (`simple_active_refine/rule_generator.py`)

**役割**: LLMを用いた情報取得ルールの生成と更新

**主な機能**:
- LLM（GPT-4o）によるHornルールの生成
- AMIE+抽出ルールを参考情報として活用
- スコア変化に基づくルールの更新
- 出力の構造化（Pydanticモデルによる厳密なスキーマ制約）

**入力**:
- 知識グラフ名
- 対象リレーション
- 生成するルール数
- （オプション）参照ルール（AMIE+抽出）
- （オプション）前イテレーションのスコア変化

**出力**:
- AmieRuleオブジェクトのリスト

**主要メソッド**:
- `generate_rules()`: 初期ルールの生成
- `update_rules()`: スコア変化に基づくルールの更新

**プロンプト設計**:
- Hornルール形式の厳密な指定
- 対象リレーションをbodyに含めない制約
- 高精度・解釈可能性を重視したルール生成の指示

**備考**:
- リトライ機構（指数バックオフ）により安定した実行を実現
- 構造化出力により、パース エラーを最小化

---

### 4. TextAttributedKnowledgeGraph (`simple_active_refine/knoweldge_retriever.py`)

**役割**: テキスト属性付き知識グラフの管理と検索

**主な機能**:
- 知識グラフの初期化と永続化（pickle + ChromaDB）
- エンティティ/リレーションのテキスト属性管理
- ベクトル検索（OpenAI embeddings）
- キーワード検索（BM25）
- ハイブリッド検索（ベクトル + キーワード）
- エンティティ/トリプルの追加と自動インデックス更新

**入力**:
- トリプルデータディレクトリ（train.txt等）
- エンティティ/リレーションのテキスト記述ファイル

**出力**:
- Entity/Tripleオブジェクトのリスト

**主要メソッド**:
- `search_entities_by_text()`: テキストによるエンティティ検索
- `find_similar_entities()`: Entityインスタンスに基づく類似検索
- `add_entity()`, `add_triple()`: 新規データの追加
- `get_all_entities()`, `get_all_triples()`: 全データの取得

**備考**:
- キャッシュ機構により2回目以降の読み込みを高速化
- 型安全なデータアクセス（Pydanticモデル）

---

### 5. LLMKnowledgeRetriever (`simple_active_refine/knoweldge_retriever.py`)

**役割**: Hornルールに基づく外部情報の取得

**主な機能**:
- GPT-4oのWeb検索機能を用いた外部情報取得
- Hornルールのbodyパターンを満たすエンティティの発見
- 構造化された知識の抽出（Entity/Tripleオブジェクト）
- ソース情報の記録

**入力**:
- 対象トリプル（head）
- Hornルール（bodyパターン）
- TextAttributedKnowledgeGraph（既存知識の参照用）

**出力**:
- RetrievedKnowledgeオブジェクト（エンティティとトリプルのリスト）

**主要メソッド**:
- `retrieve_knowledge_for_triple()`: 単一トリプルに対する情報取得
- `retrieve_knowledge_for_triples()`: 複数トリプルに対するバッチ処理

**プロンプト設計**:
- Hornルールの説明
- 既知変数の記述提供
- 構造化出力（JSON）の要求
- Web検索の効果的活用の指示

**備考**:
- OpenAI Responses APIのweb_search_preview機能を使用
- リトライ機構により安定性を確保

---

### 6. TriplesEditor (`simple_active_refine/triples_editor.py`)

**役割**: ルールに基づくトリプルの追加処理

**主な機能**:
- Hornルールのbodyパターンマッチング
- 変数の統一（unification）
- 外部取得情報との照合
- トリプルの追加とデータセット更新
- 可視化（グラフ構造）

**入力**:
- 対象トリプルリスト
- Hornルール（AmieRules）
- 外部取得情報（候補トリプル）
- 既存知識グラフ

**出力**:
- 更新されたトリプルデータセット
- 追加トリプルの記録（JSON）

**主要関数**:
- `add_triples_based_on_rules()`: メイン処理
- `find_body_triples_for_head()`: ルールbodyを満たすトリプル発見
- `_unify()`: 変数の統一処理

**備考**:
- 複数ルールの並列適用
- 重複チェックによりトリプルの一貫性を保証

---

### 7. ScoreVariationAnalyzer (`simple_active_refine/analyzer.py`)

**役割**: イテレーション間のスコア変化の分析とレポート生成

**主な機能**:
- イテレーション前後のスコア比較
- スコア変化の統計分析
- Markdown形式のレポート生成
- 可視化（スコア分布、変化量等）

**入力**:
- 現イテレーションのディレクトリ
- 次イテレーションのディレクトリ
- ルール情報

**出力**:
- スコア変化データフレーム
- Markdownレポート
- 可視化図（PNG）

**主要メソッド**:
- `create_report()`: レポート生成
- `_create_summary_table()`: サマリーテーブル作成
- `_create_detailed_analysis()`: 詳細分析セクション作成

**備考**:
- ルールごとのスコア変化分析
- 統計的有意性の検討

---

### 8. AMIE+ Integration (`simple_active_refine/amie.py`)

**役割**: AMIE+ツールとの連携とHornルールの管理

**主な機能**:
- AMIE+の実行ラッパー
- Hornルールのパース（CSV形式）
- ルールの永続化（pickle, CSV）
- ルールの品質指標管理（support, confidence, head coverage等）

**クラス構造**:
- `TriplePattern`: トリプルパターン（変数を含む）
- `AmieRule`: 単一Hornルール
- `AmieRules`: ルール集合

**主要メソッド**:
- `execute_amie()`: AMIE+の実行
- `AmieRules.from_csv()`: CSVからのルール読み込み
- `AmieRules.from_pickle()`: pickleからのルール読み込み
- `filter_by_quality()`: 品質指標によるフィルタリング

**備考**:
- DataFrame形式での操作をサポート
- 品質指標による柔軟なフィルタリング

---

## メイン処理フロー (`main.py`)

メインスクリプトは上記エージェントを統合し、反復的な改善プロセスを実行します。

### 処理ステップ

```
for i in 1 to n_iter:
    1. KnowledgeGraphEmbedding.train_model()
       └─> 知識グラフ埋込モデルの学習
    
    2. RuleExtractor.extract_rules_from_high_score_triples()
       └─> 高スコアトリプルからAMIE+ルールを抽出
    
    3. BaseRuleGenerator.generate_rules() or update_rules()
       └─> LLMによる情報取得ルール生成/更新
       └─> (初回: generate_rules, 2回目以降: update_rules with score feedback)
    
    4. LLMKnowledgeRetriever.retrieve_knowledge_for_triples()
       └─> 外部情報源から知識を取得
    
    5. TriplesEditor.add_triples_based_on_rules()
       └─> ルールに基づいてトリプルを追加
    
    6. ScoreVariationAnalyzer.create_report()
       └─> スコア変化を分析しレポート生成
```

### ディレクトリ構造

```
experiments/{date}/
├── iter-1/
│   ├── data/              # 初期データセット
│   ├── model/             # 学習済み埋込モデル
│   ├── rules/             # 抽出/生成されたルール
│   ├── updated_data/      # トリプル追加後のデータセット
│   ├── evaluations/       # 評価結果
│   └── report.md          # スコア変化レポート
├── iter-2/
│   └── ...
└── ...
```

---

## 設計原則

### コーディング規約

- **PEP8準拠**: すべてのPythonコードはPEP8スタイルガイドに従う
- **Google Style Docstring**: すべての関数・クラスにGoogle形式のdocstringを記述
- **ロギング**: `util.get_logger()`を使用した統一的なログ出力
- **可読性重視**: 変数名・関数名は意味が明確になるよう命名
- **型ヒント**: 関数の引数・戻り値には型ヒントを記述

### アーキテクチャ原則

- **単一責任の原則**: 各モジュールは明確に定義された単一の責任を持つ
- **疎結合**: モジュール間の依存関係を最小限に抑える
- **再利用性**: 汎用的な機能は適切に抽象化し再利用可能にする
- **拡張性**: 新しい埋込モデル、ルール抽出手法等を容易に追加可能な設計
- **テスタビリティ**: 各モジュールは独立してテスト可能な設計

### テスト

- **テストコードの配置**: すべてのテストは`tests/`ディレクトリに配置
- **テストカバレッジ**: 主要機能には対応するテストコードを作成
- **命名規則**: テストファイルは`test_*.py`の形式で命名
- **テスト結果の管理**: テストが成功した場合は、`tests/`ディレクトリの下に、日付と日時を名称に含むディレクトリを作成し、その中にテスト結果のレポートを作成

---

## 外部依存関係

### Python パッケージ

- **PyKEEN**: 知識グラフ埋込モデルの学習
- **LangChain**: LLMとの対話とプロンプト管理
- **OpenAI**: GPT-4o API、埋め込みAPI、Web検索API
- **ChromaDB**: ベクトルデータベース
- **rank-bm25**: キーワード検索
- **Pydantic**: データ検証と構造化出力

### 外部ツール

- **AMIE+**: Hornルールの抽出（JavaベースのCLIツール）
  - 設定: `settings.PATH_AMIE_JAR`で指定

### API キー

- **OpenAI API Key**: `settings.OPENAI_API_KEY`で設定
  - 必要な権限: GPT-4o、embeddings、web search preview

---

## ユーティリティ (`simple_active_refine/util.py`)

### 主要機能

- `get_logger()`: 統一されたログ出力設定
- その他の共通ユーティリティ関数

---

## 今後の拡張性

### 対応可能な拡張

1. **新しい埋込モデル**: PyKEENがサポートする任意のモデルを使用可能
2. **別のLLM**: LangChainを通じて他のLLMプロバイダーへの切り替えが容易
3. **外部情報源**: Web検索以外の情報源（データベース、API等）の追加
4. **ルール抽出手法**: AMIE+以外の手法の組み込み
5. **評価指標**: 新しい品質指標の追加

### 検討事項

- バッチ処理の並列化による高速化
- 段階的な検証機構の追加（追加前のトリプル検証）
- ルール品質のオンライン学習
- マルチリレーション対応の強化

---

## 関連ドキュメント

- [README.md](./README.md): プロジェクト概要と使用方法

---

## 連絡先・貢献

このプロジェクトへの貢献やフィードバックは歓迎します。Issue・Pull Requestをお気軽にお送りください。
