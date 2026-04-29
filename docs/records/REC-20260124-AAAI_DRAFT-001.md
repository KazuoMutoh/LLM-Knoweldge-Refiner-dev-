# REC-20260124-AAAI_DRAFT-001: AAAI原稿（背景・課題・目的）草稿の作成

作成日: 2026-01-24
最終更新日: 2026-01-24

## 概要
AAAI投稿を想定した論文原稿のうち、まず「背景・課題・目的（＋貢献の整理）」を日本語でLaTeX草稿として作成した。
本草稿は内容整理を目的とし、AAAI公式スタイル（aaaiXX.sty）への移行と英文化は後続作業とする。

## 成果物
- 草稿（LaTeX）: [docs/output/aaai_draft_20260124_ja.tex](../output/aaai_draft_20260124_ja.tex)
- 草稿（Markdown）: [docs/output/aaai_draft_20260124_ja.md](../output/aaai_draft_20260124_ja.md)

## 参照したドキュメント
ユーザ指定:
- [docs/external/研究の狙い.md](../external/研究の狙い.md)
- [docs/external/先行研究調査まとめ_2023.tex](../external/先行研究調査まとめ_2023.tex)
- [docs/rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md](../rules/RULE-20260111-COMBO_BANDIT_OVERVIEW-001.md)

リンク先（主要）:
- [docs/rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md](../rules/RULE-20260117-ARM_PIPELINE_IMPL-001.md)
- [docs/rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md](../rules/RULE-20260117-RUN_FULL_ARM_PIPELINE-001.md)
- [docs/rules/RULE-20260118-RELATION_PRIORS-001.md](../rules/RULE-20260118-RELATION_PRIORS-001.md)
- [docs/rules/RULE-20260119-TAKG_KGFIT-001.md](../rules/RULE-20260119-TAKG_KGFIT-001.md)
- [docs/external/KGEフレンドさを考慮したwitness評価の改善.md](../external/KGEフレンドさを考慮したwitness評価の改善.md)

## 次の作業候補
- AAAIの導入章として自然になるよう、関連研究（HITL/Active Learning/Bandit、Rule mining、TAKG/KG-FIT）を追加。
- 提案手法の章（arm生成・選択・取得・代理評価・更新・最終再学習/評価）を、実装仕様に整合させて記述。
- 英文化とAAAIテンプレへの移植（引用とbib整備を含む）。
