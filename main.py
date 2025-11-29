
import json
import os

from simple_active_refine.rule_generator import BaseRuleGenerator
from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.triples_editor import add_triples_based_on_rules
from simple_active_refine.analyzer import ScoreVraiationAnalyzer
from simple_active_refine.rule_extractor import extract_rules_from_high_score_triples
from simple_active_refine.util import get_logger
import shutil

logger = get_logger('main')

def save_markdown(md_text:str, file_path:str):
    with open(file_path, 'w') as fout:
        fout.write(md_text)


if __name__ == '__main__':
    """parameters"""
    # target
    knoweldge_graph = 'FB15k-237'
    target_relation = '/location/location/contains'
    
    # overall control
    ## a direcotory for initial triples
    ## which is assumed to have target_triples.txt, train_removed.txt and train.txt
    dir_initial_triples = './experiments/test_data_for_locations_contain'
    n_iter = 10
    dir_working  = './experiments/20251105/try7'
    n_rules = 3

    use_amie_rules = True
    min_head_converage=0.01
    min_pca_conf = 0.1
    k_neighbor = 1
    lower_percentile = 80

    n_add= 10
    
    # initial rules
    f_initial_rules = None

    # embedding
    f_config_embedding = "./config_embeddings.json"
    num_epochs = 100
    """"""

    # create workind directory
    if not os.path.exists(dir_working):
        os.makedirs(dir_working, exist_ok=True)
    
    # read common parameter settings
    with open(f_config_embedding) as fin:
        config_embedding = json.load(fin)
    config_embedding["training_kwargs"]["num_epochs"] = num_epochs

    with open(os.path.join(dir_initial_triples,'config_dataset.json'), 'r') as fin:
        config_dataset = json.load(fin)


    logger.info('**Step.0 Generate initial rules & learn initial knowledge graph embeddings.**')
    # copy dir_initial_triples to dir_working and rename to iter_1
    dir_current_iter = os.path.join(dir_working,'iter_1')
    if os.path.exists(dir_current_iter):
        logger.info(f'removing exsiting directory {dir_current_iter}')
        shutil.rmtree(dir_current_iter)
    shutil.copytree(dir_initial_triples, dir_current_iter)
    
    # learn initial knoweldge graph embeddings
    config_embedding['dir_triples'] = dir_current_iter
    config_embedding['dir_save'] = dir_current_iter
    kge = KnowledgeGraphEmbedding.train_model(**config_embedding)

    if use_amie_rules:
        amie_rules = extract_rules_from_high_score_triples(kge, 
                                                           config_dataset['target_relation'],
                                                           min_pca_conf=min_pca_conf,
                                                           min_head_coverage=min_head_converage,
                                                           k_neighbor=k_neighbor,
                                                           lower_percentile=lower_percentile)
        amie_rules.to_csv(os.path.join(dir_current_iter, 'amie_rules.csv'))
    else:
        amie_rules = None

    # generate initial rules
    if f_initial_rules is None:
        rule_generator = BaseRuleGenerator()
        rules = rule_generator.generate_rules(knoweldge_graph, target_relation, n_rules=3)
    
    rules.to_pickle(os.path.join(dir_current_iter, 'rules.pkl'))
    rules.to_csv(os.path.join(dir_current_iter, 'rules.csv'))
    
    for i in range(1, n_iter):
        
        logger.info(f'---start {i} th iteration.---')
        
        logger.info('**Step.1 Add triples based on rules**')
        logger.info(f'current rules:{rules.rules}')
        dir_next_iter = os.path.join(dir_working, f'iter_{i+1}')
        if os.path.exists(dir_next_iter):
            logger.info(f'removing exsiting directory {dir_next_iter}')
            shutil.rmtree(dir_next_iter)

        add_triples_based_on_rules(dir_triples=dir_current_iter, 
                                dir_updated_triples=dir_next_iter,
                                rules=rules,
                                n_add=n_add
                                )
    
        logger.info('**Step.2 Update knoweldge graph embedding**')
        config_embedding['dir_triples'] = dir_next_iter
        config_embedding['dir_save'] = dir_next_iter
        kge = KnowledgeGraphEmbedding.train_model(**config_embedding)
        
        logger.info('**Step.3 Analyze score variation.**')
        analyzer = ScoreVraiationAnalyzer(dir_iter=dir_current_iter,
                                          dir_next_iter=dir_next_iter,
                                          rules=rules)
        report = analyzer.generate_report_for_score_variations()
        save_markdown(report,
                      os.path.join(dir_next_iter,'report_score_variation_analysis.md'))
        summary_report = analyzer.generate_report(os.path.join(dir_next_iter, 'report_score_variation_analysis_summary.md'))
        analyzer.df_score_diff.to_pickle(os.path.join(dir_next_iter, 'df_score_diff.pkl'))
        
        logger.info('**Step.4 Update rules based on analysis.**')
        if use_amie_rules:
            amie_rules = extract_rules_from_high_score_triples(kge, 
                                                               config_dataset['target_relation'],
                                                               min_pca_conf=min_pca_conf,
                                                               min_head_coverage=min_head_converage,
                                                               k_neighbor=k_neighbor,
                                                               lower_percentile=lower_percentile)
            amie_rules.to_csv(os.path.join(dir_next_iter, 'amie_rules.csv'))
        else:
            amie_rules = None

        updated_rules, observation, thought = rule_generator.update_rules(rules, report, amie_rules)
        logger.info(type(observation))
        logger.info(f'observation:{observation}')
        logger.info(type(thought))
        logger.info(f'thought:{thought}')
        logger.info(f'updated rules:{updated_rules.rules}')

        rules_update_report = (
            '# rules update report\n'
            f'## observation\n{observation}\n\n'
            f'## thought\n{thought}\n\n'
            f'## updated rules\n{updated_rules.to_markdown_list()}\n'
        )

        save_markdown(rules_update_report, os.path.join(dir_next_iter, 'report_rules_update.md'))

        logger.info('**Step.5 Save results.**')
        updated_rules.to_pickle(os.path.join(dir_next_iter, 'rules.pkl'))
        updated_rules.to_csv(os.path.join(dir_next_iter, 'rules.csv'))

        # update
        rules = updated_rules
        dir_current_iter = dir_next_iter

        

    


        
        
        

