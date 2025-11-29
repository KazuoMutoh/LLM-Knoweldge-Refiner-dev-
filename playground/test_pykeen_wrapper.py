import sys
import pylab as plt
sys.path.append('/home/acg16558pn/programs/Simple-Active-Refinement-for-Knowledge-Graph')
from simple_active_refine.embedding import KnowledgeGraphEmbedding

model_dir = 'models/20250903/pykeen_test_transe'
kge = KnowledgeGraphEmbedding(model_dir=model_dir)

labeled_triples = kge.get_labeled_triples()
triples = kge._label_to_id(labeled_triples)
scores = kge.score_triples(labeled_triples=labeled_triples)
norm_socres = kge.score_triples(labeled_triples=labeled_triples, normalize=True)    

plt.figure(figsize=(8,6))
plt.hist(scores, bins=50, alpha=0.5, label='Raw Scores')
plt.savefig('fig_raw_scores.png')