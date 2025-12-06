from __future__ import annotations
import os
import json
import argparse
import pandas as pd
import pylab as plt
from dataclasses import dataclass
from typing import List, Tuple

from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.amie import AmieRules
from simple_active_refine.util import get_logger
from simple_active_refine.rule_history import RuleEvaluationRecord


logger = get_logger('analyzer')

class ScoreVraiationAnalyzer:
    """
    知識グラフのスコア変化を解析するためのクラス
    """
    def __init__(self, 
                 dir_iter, 
                 dir_next_iter, 
                 iter_name='i',
                 next_iter_name='i+1',
                 f_rules=None, 
                 rules:AmieRules=None):
        
        self.iter_name = iter_name
        self.next_iter_name = next_iter_name

        logger.info('start creating report.')
        logger.info('read triples and rules')

        self.target_triples = self._read_tsv(os.path.join(dir_iter,
                                                          'target_triples.txt'))
        
        self.triples = self._read_tsv(os.path.join(dir_iter, 'train.txt'))
        self.updated_triples = self._read_tsv(os.path.join(dir_next_iter, 'train.txt'))

        with open(os.path.join(dir_next_iter, 'added_triples_by_target.json'), 'r') as fin:
            _added_triples_by_target = json.load(fin)
            self.added_triples_by_target = dict()
            for dict_data in _added_triples_by_target:
                taget_triple = tuple(dict_data['target_triple'])
                added_triples = [tuple(_) for _ in dict_data['triples_to_be_added']]
                self.added_triples_by_target[taget_triple] = added_triples

        if rules is not None:
            self.rules = rules
        else:
            if f_rules is None:
                f_rules = os.path.join(dir_iter, 'extracted_rules.pkl')
                self.rules = AmieRules.from_pickle(f_rules)
            else:
                if f_rules.endswith('.pkl'):
                    self.rules = AmieRules.from_pickle(f_rules)
                elif f_rules.endswith('.csv'):
                    self.rules = AmieRules.from_csv(f_rules)
                else:
                    raise Exception('f_rules should be .pkl or .csv')          
        
        logger.info('read triples and rules')
        with open(os.path.join(dir_iter, 'config_dataset.json'), 'r') as fin:
            self.config_dataset = json.load(fin)
        
        logger.info('read knoweldge graph embeddings')
        self.kge = KnowledgeGraphEmbedding(model_dir=dir_iter)
        self.updated_kge = KnowledgeGraphEmbedding(model_dir=dir_next_iter)

        logger.info('calculate score of target tripels')
        self.scores = self.kge.score_triples(self.target_triples)
        self.updated_scores = self.updated_kge.score_triples(self.target_triples)

        logger.info('make a table for score variation')
        list_table = []
        for (h,r,t), score, updated_score in zip(self.target_triples, 
                                                 self.scores, 
                                                 self.updated_scores):
            
            dict_row = {'head': h, 'relation':r, 'tail': t, 
                        f'score ({self.iter_name})': score, 
                        f'score ({self.next_iter_name})':updated_score,
                        f'score diff ({self.next_iter_name}-{self.iter_name})': updated_score - score,
                        f'num. of added triples': len(self.added_triples_by_target.get((h,r,t), []))
                        }
            list_table.append(dict_row)  
        self.df_score_diff = pd.DataFrame(list_table)


    @staticmethod
    def _to_str_triple(tuple_triple):
        return '\t'.join(tuple_triple)

    @staticmethod
    def _read_tsv(filepath):
        list_data = []
        with open(filepath, 'r') as fin:
            for row in fin:
                h, r, t = row.rstrip().split('\t')
                list_data.append((h,r,t))
        return list_data
    
    def generate_report_for_score_variations(self):
        """
        LLMに与えるためのフィードバックを生成する
        """

        feedback = "The following is a report on score variations for knowledge graph triples.\n"
        feedback += "For each triple, the number of added triples and score changes are shown below.\n\n"
        
        for i, target_triple in enumerate(self.target_triples):
            
            h, r, t = target_triple
            list_added_triples = self.added_triples_by_target.get(target_triple, [])
            num_added = len(list_added_triples)

            if num_added == 0:
                continue

            score_before = self.scores[i]
            score_after = self.updated_scores[i]
            score_diff = score_after - score_before
            
            feedback += 30*'-' + '\n'
            feedback += f"target triple: ({h}, {r}, {t})\n"
            feedback += f"number of added triples: {num_added}\n"
            feedback += f"score before update: {score_before}\n"
            feedback += f"score after update: {score_after}\n"
            feedback += f"score variation: {score_diff}\n"
            feedback += "added triples:\n"
            for added_triple in list_added_triples:
                ah, ar, at = added_triple
                feedback += f"- ({ah}, {ar}, {at})\n"
            feedback += '\n'

        return feedback

    def generate_report(self, filepath, title='Experiments'):
        """
        triplesのスコア変化に関するレポートを生成するメソッド.（人間向け）
        """

        logger.info('start generating report')

        
        md_text = f'# {title}\n'
        
        md_text += f'## Target Triples\n'
        for triple in self.target_triples:
            md_text += f'- {triple}\n'
        
        md_text += f'## Rules\n'
        md_text += self.rules.to_dataframe()[['body','head']].to_markdown(index=False)
        md_text += '\n'
        
        md_text += f'## Score difference\n'

        mean_score_diff = self.df_score_diff[self.df_score_diff['num. of added triples']>0][f'score diff ({self.next_iter_name}-{self.iter_name})'].mean()
        md_text += f'**Mean score difference (only added triples): {mean_score_diff}**\n\n'

        md_text += self.df_score_diff.to_markdown()

        self.df_score_diff.to_csv(os.path.splitext(filepath)[0]+'.csv', index=False)
        
        fig, ax = plt.subplots(nrows=2, figsize=(6,4),sharex=True)
        print(self.df_score_diff[self.df_score_diff.columns[0]])
        self.df_score_diff[self.df_score_diff.columns[3]].plot(kind='hist', ax=ax[0], title='Score distribution before update', bins=50)
        ax[0].set_xlabel('Score')
        self.df_score_diff[self.df_score_diff.columns[4]].plot(kind='hist', ax=ax[1], title='Score distribution after update', bins=50)
        ax[0].set_xlabel('Score')
        plt.tight_layout()
        fig_path = os.path.splitext(filepath)[0]+'.png'
        plt.savefig(fig_path)
        plt.close(fig)

        fig, az = plt.subplots(nrows=1, figsize=(6,4))
        num_added_triples = self.df_score_diff[self.df_score_diff.columns[6]]
        diff_scores = self.df_score_diff[self.df_score_diff.columns[5]]
        az.scatter(num_added_triples, diff_scores, marker='.')
        az.set_xlabel('Number of added triples')
        az.set_ylabel('Score difference')
        az.set_title('Score difference vs. Number of added triples')
        plt.tight_layout()
        fig_path = os.path.splitext(filepath)[0]+'_scatter.png'
        plt.savefig(fig_path)
        plt.close(fig)

        with open(filepath, 'w') as fout:
            fout.write(md_text)

        return md_text


class RuleWiseAnalyzer:
    """ルールごとの独立した評価を行うアナライザー
    
    単一ルールが特定のtarget tripleセットに対してトリプルを追加した結果、
    スコアがどのように変化したかを評価し、RuleEvaluationRecordを生成する。
    """
    
    def __init__(self, kge_before: KnowledgeGraphEmbedding, kge_after: KnowledgeGraphEmbedding):
        """RuleWiseAnalyzerの初期化
        
        Args:
            kge_before: トリプル追加前の埋込モデル
            kge_after: トリプル追加後の埋込モデル
        """
        self.kge_before = kge_before
        self.kge_after = kge_after
    
    def create_evaluation_record(self,
                                 iteration: int,
                                 rule_id: str,
                                 rule,
                                 target_triples: List[Tuple],
                                 added_triples: List[Tuple]) -> RuleEvaluationRecord:
        """ルールの評価記録を作成
        
        Args:
            iteration: iteration番号
            rule_id: ルールID
            rule: AmieRuleオブジェクト
            target_triples: 対象となったトリプルのリスト
            added_triples: 追加されたトリプルのリスト
        
        Returns:
            RuleEvaluationRecord: 評価記録
        """
        # トリプル追加前後のスコアを計算
        scores_before = self.kge_before.score_triples(target_triples)
        scores_after = self.kge_after.score_triples(target_triples)
        
        # スコア変化の計算
        score_changes = [after - before for before, after in zip(scores_before, scores_after)]
        
        # 統計値の計算
        import statistics
        mean_score_change = statistics.mean(score_changes) if score_changes else 0.0
        std_score_change = statistics.stdev(score_changes) if len(score_changes) > 1 else 0.0
        
        positive_changes = sum(1 for sc in score_changes if sc > 0)
        negative_changes = sum(1 for sc in score_changes if sc < 0)
        
        logger.info(f"Rule {rule_id}: mean_Δ={mean_score_change:.6f}, "
                   f"pos={positive_changes}, neg={negative_changes}, "
                   f"added={len(added_triples)}")
        
        return RuleEvaluationRecord(
            iteration=iteration,
            rule_id=rule_id,
            rule=rule,
            target_triples=target_triples,
            added_triples=added_triples,
            score_changes=score_changes,
            mean_score_change=mean_score_change,
            std_score_change=std_score_change,
            positive_changes=positive_changes,
            negative_changes=negative_changes
        )
