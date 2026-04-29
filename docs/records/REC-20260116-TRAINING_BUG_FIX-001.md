# REC-20260116-TRAINING_BUG_FIX-001: KGE初期モデル学習時のパラメータ名エラー修正

作成日: 2026-01-16
参照: [REC-20260116-FULL_PIPELINE_EXPERIMENT-001](REC-20260116-FULL_PIPELINE_EXPERIMENT-001.md)
関連ルール: [RULE-20260111-COMBO_BANDIT_OVERVIEW-001](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)

---

## 概要

実験用初期KGEモデルの学習を開始しようとした際、`python`コマンド不在と`num_epochs`パラメータ名の誤りにより学習が失敗した。本記録はその原因と対策を記載する。

## 発生した問題

### 問題1: `python`コマンド不在

**症状**:
```bash
bash: python: command not found
```

**原因**:
- Dev container環境では`python3`コマンドのみが利用可能
- `python`エイリアスが設定されていない

**対策**:
- すべての実行コマンドを`python`から`python3`に変更
- 今後の実験スクリプトでも`python3`を使用することを徹底

### 問題2: `kge_trainer_impl.py`にCLIエントリポイント不在

**症状**:
- `python3 -m simple_active_refine.kge_trainer_impl`を実行しても何も起こらない（出力なし）

**原因**:
- `simple_active_refine/kge_trainer_impl.py`は`FinalKGETrainer`クラスのみを提供するモジュールで、`if __name__ == "__main__"`ブロックが存在しない
- このファイルはパイプライン内部で使用されることを想定しており、単独実行用には設計されていない

**対策**:
- 独立した学習スクリプト`scripts/train_initial_kge.py`を新規作成
- このスクリプトは`KnowledgeGraphEmbedding.train_model()`を直接呼び出す

### 問題3: `num_epochs`パラメータ名の誤り

**症状**:
```
TypeError: pipeline() got an unexpected keyword argument 'num_epochs'
```

**原因**:
- PyKEENの`pipeline()`関数は`num_epochs`ではなく`epochs`という引数名を使用
- `simple_active_refine/embedding.py`の`train_model()`メソッドは`**pipeline_kwargs`として引数を転送するため、正しい引数名を使用する必要がある

**検証**:
```python
from pykeen.pipeline import pipeline
import inspect
sig = inspect.signature(pipeline)
print([p for p in sig.parameters.keys()])
# 結果に 'epochs' は含まれるが 'num_epochs' は含まれない
```

**対策**:
- `scripts/train_initial_kge.py`で`num_epochs`を`epochs`に修正:
  ```python
  kge = KnowledgeGraphEmbedding.train_model(
      model=model,
      dir_triples=args.dir_triples,
      dir_save=args.output_dir,
      epochs=args.num_epochs,  # ← num_epochsではなくepochs
      **config
  )
  ```

## 作成したファイル

### `/app/scripts/train_initial_kge.py`

初期KGEモデル学習用の独立スクリプト。以下の機能を提供:

- コマンドライン引数:
  - `--dir_triples`: 学習データディレクトリ（train.txt, valid.txt, test.txt）
  - `--output_dir`: 学習済みモデルの保存先
  - `--embedding_config`: 埋め込み設定JSONファイル（config_embeddings.json等）
  - `--num_epochs`: 学習エポック数（デフォルト: 100）

- 機能:
  1. config JSONから設定を読み込み
  2. `model`パラメータを抽出（`TransE`等）
  3. `KnowledgeGraphEmbedding.train_model()`を呼び出し
  4. 学習完了後に評価メトリクスを計算・保存（`initial_metrics.json`）

## 実行結果

### 成功した学習コマンド

```bash
cd /app && python3 scripts/train_initial_kge.py \
  --dir_triples /app/experiments/test_data_for_nationality_v3 \
  --output_dir /app/models/20260116/fb15k237_transe_nationality \
  --embedding_config config_embeddings.json \
  --num_epochs 100 \
  2>&1 | tee /app/models/20260116/training_output.log
```

### 学習進行状況（2026-01-16 23:47時点）

- モデル: TransE
- データセット: test_data_for_nationality_v3 (204 triples after filtering)
- デバイス: cuda:0 (GPU)
- 進行: 4/100 エポック完了
- 予想所要時間: 約50分

### 学習設定

```json
{
  "model": "transe",
  "model_kwargs": {
    "embedding_dim": 64,
    "scoring_fct_norm": 1
  },
  "loss": "CrossEntropyLoss",
  "training_loop": "lcwa",
  "stopper": null,
  "optimizer": "adam",
  "optimizer_kwargs": {
    "lr": 0.0016608460884079603,
    "weight_decay": 0.0
  },
  "training_kwargs": {
    "batch_size": 256,
    "label_smoothing": 0.717650072390557
  },
  "evaluator": "RankBasedEvaluator",
  "random_seed": 42
}
```

## 教訓と今後の対策

### 1. PyKEEN APIの正確な理解

- PyKEENの`pipeline()`関数は`epochs`引数を使用（`num_epochs`ではない）
- 他のフレームワークとの混同に注意
- 不明な引数名は`inspect.signature()`で確認

### 2. 既存コード優先の原則

- ユーザーからの指示通り、新規実装前に既存コードを十分に確認すべきだった
- `simple_active_refine/pipeline_concrete.py`の`PyKeenKGETrainerV3`が既に正しい実装を持っていた（L311: `num_epochs=self.num_epochs`）
- しかし、実際は`PyKeenKGETrainerV3.train_and_evaluate()`内で`epochs`として渡している
- 今後は既存実装をテンプレートとして活用

### 3. スクリプトのテスト

- 新規作成したスクリプトは小規模データで動作確認後に本実行すべき
- 今回は初回実行で3回のエラー修正が必要だった（python → python3 → num_epochs → epochs）

### 4. ドキュメント参照の重要性

- [RULE-20260111-COMBO_BANDIT_OVERVIEW-001](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)には実装状況が記載されていた
- 特に「既存の実装」セクションを参照すべきだった
- 今後は設計ルール/標準の参照を徹底

## 次のステップ

1. ✅ 初期KGEモデルの学習（進行中、約45分残り）
2. ⏳ 学習完了後、実験Aの実行開始
3. ⏳ 実験B・Cシリーズの順次実行
4. ⏳ 結果の分析とレポート作成

---

更新日: なし（初版）
