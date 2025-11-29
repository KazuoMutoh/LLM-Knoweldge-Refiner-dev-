import json

f_entity = './processed_data/20250826/fb15k-237_drop_nationality/selected_targets.txt'
f_triple = './processed_data/20250826/fb15k-237_drop_nationality/train.txt'
target_relation = '/people/person/nationality'

# selected_targets.txtから対象エンティティを読み込む
with open(f_entity, 'r') as f:
    target_entities = set(line.strip() for line in f if line.strip())

target_triples = []

# train.txtからtripleを抽出
with open(f_triple, 'r') as f:
    for line in f:
        head, relation, tail = line.strip().split('\t')
        if head in target_entities and relation == target_relation:
            target_triples.append({'head': head, 'relation': relation, 'tail': tail})


print(target_triples)

# target_triples.jsonに保存
with open('./processed_data/20250826/fb15k-237_drop_nationality/target_triples.json', 'w') as f:
    json.dump(target_triples, f, ensure_ascii=False, indent=2)