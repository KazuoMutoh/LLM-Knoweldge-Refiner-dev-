# KG-FIT

KG-FIT（Jiang et al., 2024; arXiv:2405.16412）は、LLM/PLMをファインチューニングせずに、
- テキスト埋め込み（グローバル意味）
- LLM誘導で得たエンティティ階層（グローバル構造）
- KGリンク予測（ローカル意味）

を **KGE側の学習目標に統合**して、リンク予測性能を向上させるフレームワーク。

- 論文: https://arxiv.org/abs/2405.16412
- GitHub: https://github.com/pat-jj/KG-FIT

## 何をしているか（要約）

### 1) 事前計算: テキスト埋め込み
- 各エンティティの名前/説明文から埋め込みを作り、結合して初期表現を作る。

### 2) 事前計算: 階層（クラスタ階層）の構築
- agglomerative clustering で seed 階層を作り、任意でLLMにより階層を精錬（LHR）。

### 3) KGEの微調整
- ベースKGE（TransE/RotatE/HAKE等）を維持しつつ、
  - リンク予測損失
  - 階層制約（クラスタ凝集/分離/階層距離維持）
  - セマンティック・アンカー制約（テキスト埋め込みからの逸脱抑制）
  を同時に最適化する。

## 本プロジェクトでの位置づけ

- TAKG対応のKGEとして、STAGEではなくKG-FITを採用する。
- OpenAI `text-embedding-3-small` を用いた **事前計算・キャッシュ**を前提にし、学習中のAPI呼び出しは避ける。

詳細な統合方針は以下の記録を参照。
- [docs/records/REC-20260119-TAKG_KGFIT-001.md](../records/REC-20260119-TAKG_KGFIT-001.md)
