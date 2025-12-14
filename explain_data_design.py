#!/usr/bin/env python3
"""データ設計の詳細説明：なぜ無関係な第三者のトリプルを削除するのか"""

import json

def main():
    print("="*100)
    print("【データ設計の詳細説明】")
    print("="*100)
    print()
    
    print("## 3種類のエンティティ")
    print()
    print("1. ターゲットエンティティ (T)")
    print("   - スコア検証の対象となる人物")
    print("   - 例: /m/06b0d2（ある俳優）")
    print()
    print("2. 近傍エンティティ (R)")
    print("   - ターゲットの周辺情報に関わるエンティティ")
    print("   - 例: 受賞作品、教育機関、プロデュース作品")
    print("   - これらに関連するトリプルを削除 → D1（ローカルな文脈の削除）")
    print()
    print("3. 無関係な第三者 (O)")
    print("   - ターゲットとも近傍とも関係のないエンティティ")
    print("   - 例: 他の俳優、他の人物")
    print("   - これらのnationalityトリプルの一部を削除 → D2（グローバルな知識の提供）")
    print()
    
    print("="*100)
    print("【具体例】")
    print("="*100)
    print()
    
    # データを読み込み
    with open('./experiments/20251213/test_nationality_v2/entity_removed_mapping.json', 'r') as f:
        mapping = json.load(f)
    
    # サンプルターゲットエンティティ
    target = "/m/06b0d2"
    
    print(f"ターゲットエンティティ: {target}")
    print()
    
    print("【このターゲットに対する処理】")
    print()
    print("1. ターゲットのnationalityトリプル:")
    print(f"   {target} --/people/person/nationality--> /m/09c7w0")
    print("   → train.txtに保持（削除しない）")
    print("   → スコア検証の対象")
    print()
    
    if target in mapping:
        removed = mapping[target]
        print(f"2. ターゲットの周辺情報（D1）: {len(removed)}個のトリプルを削除")
        print("   - 受賞、プロデュース作品、TV出演など")
        print("   - これらのエンティティ（R）に関連するトリプルを削除")
        print("   - ローカルな文脈を削除")
        print()
    
    print("3. 無関係な第三者のnationality（D2）: 545個のトリプルを削除")
    print("   - これらは「他の人物」のnationalityトリプル")
    print("   - ターゲットとは無関係（直接的にも間接的にも）")
    print()
    
    # train_removed.txtから他人のnationalityトリプルをサンプル
    nationality_triples = []
    with open('./experiments/20251213/test_nationality_v2/train_removed.txt', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts
                if r == '/people/person/nationality':
                    nationality_triples.append((h, r, t))
    
    print("   【他人のnationalityトリプルの例】")
    for h, r, t in nationality_triples[:5]:
        print(f"   {h} --{r}--> {t}")
    print(f"   ... 他 {len(nationality_triples) - 5}個")
    print()
    
    print("="*100)
    print("【なぜ無関係な第三者の情報が必要か？】")
    print("="*100)
    print()
    
    print("Hornルールの例:")
    print("  ?f nationality ?b ∧ ?a friend ?f => ?a nationality ?b")
    print("  （友人と同じ国籍を持つ傾向がある）")
    print()
    
    print("このルールを適用するには:")
    print()
    print("1. ?a（ターゲット）の友人 ?f を特定する")
    print("   → 友人関係のトリプルは削除されている可能性が高い（D1に含まれる）")
    print("   → 外部情報源から取得する必要がある")
    print()
    print("2. ?f の nationality を知る")
    print("   → これが D2（無関係な第三者のnationality）に含まれる")
    print("   → ?f はターゲットとは別の人物")
    print()
    print("3. ルールを適用して ?a の nationality を推論")
    print()
    
    print("つまり:")
    print("  - D1（ローカルな文脈）を削除することで推論を困難にする")
    print("  - D2（グローバルな知識）を提供することでルールベースの推論を可能にする")
    print()
    
    print("="*100)
    print("【実装上の確認】")
    print("="*100)
    print()
    
    print("compute_deletions() 関数での条件:")
    print()
    print("for h, r, t in base:")
    print("    if r == target_relation:")
    print("        # ターゲットエンティティを含む → スキップ（保持）")
    print("        if (h in target_entities) or (t in target_entities):")
    print("            continue")
    print("        # 近傍エンティティを含む → スキップ（既にD1で処理済み）")
    print("        if (h in selected_entities) or (t in selected_entities):")
    print("            continue")
    print("        # ここに到達 → 無関係な第三者のトリプル")
    print("        other_target_triples.append((h, r, t))")
    print()
    print("→ この条件により、ターゲットとも近傍とも関係のない")
    print("  「完全に無関係な第三者」のトリプルのみが選択される")

if __name__ == "__main__":
    main()
