# RULE-20260123-KGFIT_PAIRRE-001: KG-FIT（PairRE）運用標準（設定・比較・受け入れ）

作成日: 2026-01-23
最終更新日: 2026-01-23

---

## 0. 目的とスコープ

本ドキュメントは、KG-FITバックエンドに追加した PairRE を、実験・運用でブレなく使えるようにするための標準（rules）を定義する。

対象:
- KG-FITバックエンドでの `model="pairre"` 運用
- TransE(KG-FIT) と PairRE(KG-FIT) の比較実験の最小要件
- 再現性のための受け入れ基準

非スコープ:
- PairRE論文の詳細
- KG-FIT（TAKG）の成果物仕様そのもの（別rulesへ委譲）

参照（既存rules）:
- KG-FITバックエンド標準: [docs/rules/RULE-20260119-TAKG_KGFIT-001.md](RULE-20260119-TAKG_KGFIT-001.md)
- KG-FIT正則化/高速化: [docs/rules/RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001.md](RULE-20260119-KGFIT_REGULARIZER_SPEEDUP-001.md)

参照（検討/実験 records）:
- 実装計画: [docs/records/REC-20260121-KGFIT_PAIRRE_IMPLEMENTATION_PLAN-001.md](../records/REC-20260121-KGFIT_PAIRRE_IMPLEMENTATION_PLAN-001.md)
- 比較実験（priors=off）: [docs/records/REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001.md](../records/REC-20260122-UCB_PRIORS_OFF_KGFIT_PAIRRE-001.md)
- ランダムベースライン（kgfit_pairre）: [docs/records/REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001.md](../records/REC-20260123-RANDOM_BASELINE_KGFIT_PAIRRE-001.md)

---

## 1. 実装上の前提（PairRE差分）

KG-FITバックエンドにおける PairRE の差分は、原則として次のみに限定する。

- entity representation: KG-FIT（テキスト埋め込み初期化 + 正則化）は TransE と同一
- relation representation: relation embedding を2本持つ（$r_h, r_t$）
- interaction: `PairREInteraction(p=scoring_fct_norm)`

注意:
- `model` 名は lower-case 化されるため `"PairRE"` / `"pairre"` は同義だが、標準として `"pairre"` を用いる
- KG-FITでは entity embedding 次元はテキスト埋め込み由来で決まるため、`model_kwargs.embedding_dim` は基本的に使わない

---

## 2. embedding config（標準形）

### 2.1 最小（推奨）

```json
{
  "model": "pairre",
  "embedding_backend": "kgfit",
  "kgfit": {
    "reshape_strategy": "full",
    "hierarchy": {"neighbor_k": 5},
    "regularizer": {
      "anchor_weight": 0.5,
      "cohesion_weight": 0.5,
      "separation_weight": 0.5,
      "separation_margin": 0.2
    }
  },
  "model_kwargs": {"scoring_fct_norm": 1}
}
```

### 2.2 既存サンプル

- `config_embeddings_kgfit_pairre_fb15k237.json`: FB15k-237用（パス固定、PairRE）

---

## 3. 比較実験の標準チェックリスト（TransE vs PairRE）

TransE(KG-FIT) と PairRE(KG-FIT) を比較する場合、次を原則として揃える。

1) データ
- `dataset_dir`（`train/valid/test`）が同一
- `.cache/kgfit/` の成果物が同一（同じテキスト埋め込みと seed 階層）

2) 学習設定
- `random_seed` を固定
- `num_epochs` を揃える
- `training_loop` を揃える（既定 `slcwa`）
- optimizer と `lr`、`batch_size` を揃える
- KG-FIT正則化の重み（anchor/cohesion/separation/margin）を揃える
- `neighbor_k` を揃える

3) モデル差分
- 差分は `model` のみ（`transe` vs `pairre`）
- `scoring_fct_norm` を明示し、同値に揃える

4) 評価
- Hits@k / MRR と target score の両方を記録する
- 「before/after再スコア」などの比較手順は run 単位で固定し、run_dirを分ける

---

## 4. 受け入れ基準（実装・再現性）

PairREをKG-FITバックエンドで使う場合、最低限次を満たすこと。

- KG-FITの成果物が揃っている（`.cache/kgfit/`）
- PairREスモークが通る（ユニットテスト）:
  - [tests/test_kgfit_pairre_backend.py](../../tests/test_kgfit_pairre_backend.py)

推奨（比較実験の安全策）:
- まず `regularizer.separation_weight=0.0` + `hierarchy.neighbor_k=0` の軽量設定で学習が走ることを確認し、その後に標準設定（neighbor_k=5, separationあり）へ戻す

---

## 変更履歴

- 2026-01-23: 新規作成（KG-FIT(PairRE)の運用標準を定義）
