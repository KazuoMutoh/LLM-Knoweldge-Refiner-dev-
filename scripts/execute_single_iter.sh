# パラメータ
dir_org_triples="./data/FB15k-237"
dir_triples="./experiments/20251008/try1/iter1"
dir_updated_triples="./experiments/20251008/try1/iter2"
target_relation="/people/person/nationality"

# テストデータの作成
python 00_make_test_triples.py --dir_triples $dir_org_triples --dir_test_triples $dir_triples --target_relation $target_relation

# iter=1
python 10_learn_knowledge_graph_embedding.py --dir_triples $dir_triples --epoch 200
python 20_extract_rules.py --dir_embedding_model $dir_triples
python 21_filter_rules.py --dir_rules $dir_triples
python 30_add_triples_based_on_rules.py --dir_triples $dir_triples --dir_updated_triples $dir_updated_triples

# iter=2
python 10_learn_knowledge_graph_embedding.py --dir_triples $dir_updated_triples --epoch 200
