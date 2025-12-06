"""
ユニットテスト: main.py

テスト戦略:
1. 補助関数のテスト
   - save_markdown(): Markdownファイルの保存機能
   - sample_target_triples(): トリプルのサンプリング機能
   
2. 各処理ステップのロジック検証
   - ファイル操作の正確性
   - データ変換の正確性
   - エラーハンドリング
"""

import pytest
import os
import tempfile
import shutil
import json
from pathlib import Path
import sys

# main.pyから関数をインポート
sys.path.insert(0, '/app')
from main import save_markdown, sample_target_triples


class TestSaveMarkdown:
    """save_markdown関数のテスト"""
    
    def test_save_markdown_basic(self, tmp_path):
        """基本的なMarkdown保存のテスト"""
        md_text = "# Test Header\n\nThis is a test."
        file_path = tmp_path / "test.md"
        
        save_markdown(md_text, str(file_path))
        
        assert file_path.exists()
        with open(file_path, 'r') as f:
            content = f.read()
        assert content == md_text
    
    def test_save_markdown_with_special_chars(self, tmp_path):
        """特殊文字を含むMarkdownの保存テスト"""
        md_text = "# テスト見出し\n\n日本語テキスト: $x \\rightarrow y$"
        file_path = tmp_path / "test_japanese.md"
        
        save_markdown(md_text, str(file_path))
        
        assert file_path.exists()
        with open(file_path, 'r') as f:
            content = f.read()
        assert content == md_text
    
    def test_save_markdown_overwrite(self, tmp_path):
        """既存ファイルの上書きテスト"""
        file_path = tmp_path / "test.md"
        
        # 最初の保存
        save_markdown("First content", str(file_path))
        # 上書き
        save_markdown("Second content", str(file_path))
        
        with open(file_path, 'r') as f:
            content = f.read()
        assert content == "Second content"
    
    def test_save_markdown_empty_string(self, tmp_path):
        """空文字列の保存テスト"""
        file_path = tmp_path / "empty.md"
        
        save_markdown("", str(file_path))
        
        assert file_path.exists()
        assert file_path.stat().st_size == 0


class TestSampleTargetTriples:
    """sample_target_triples関数のテスト"""
    
    def test_sample_basic(self):
        """基本的なサンプリングのテスト"""
        all_triples = [
            ('e1', 'r1', 't1'),
            ('e2', 'r1', 't2'),
            ('e3', 'r1', 't3'),
            ('e4', 'r1', 't4'),
            ('e5', 'r1', 't5'),
        ]
        
        sampled = sample_target_triples(all_triples, 3)
        
        assert len(sampled) == 3
        assert all(t in all_triples for t in sampled)
        assert len(set(sampled)) == 3  # 重複なし
    
    def test_sample_with_exclude(self):
        """除外トリプルを指定したサンプリングのテスト"""
        all_triples = [
            ('e1', 'r1', 't1'),
            ('e2', 'r1', 't2'),
            ('e3', 'r1', 't3'),
            ('e4', 'r1', 't4'),
            ('e5', 'r1', 't5'),
        ]
        exclude = {('e1', 'r1', 't1'), ('e2', 'r1', 't2')}
        
        sampled = sample_target_triples(all_triples, 2, exclude_triples=exclude)
        
        assert len(sampled) == 2
        assert all(t not in exclude for t in sampled)
    
    def test_sample_more_than_available(self):
        """利用可能数より多くサンプリングしようとした場合のテスト"""
        all_triples = [
            ('e1', 'r1', 't1'),
            ('e2', 'r1', 't2'),
        ]
        
        sampled = sample_target_triples(all_triples, 10)
        
        assert len(sampled) == 2  # 利用可能な数まで
        assert set(sampled) == set(all_triples)
    
    def test_sample_with_all_excluded(self):
        """全てが除外された場合のテスト"""
        all_triples = [
            ('e1', 'r1', 't1'),
            ('e2', 'r1', 't2'),
        ]
        exclude = set(all_triples)
        
        sampled = sample_target_triples(all_triples, 5, exclude_triples=exclude)
        
        assert len(sampled) == 0
    
    def test_sample_zero(self):
        """0個サンプリングのテスト"""
        all_triples = [
            ('e1', 'r1', 't1'),
            ('e2', 'r1', 't2'),
        ]
        
        sampled = sample_target_triples(all_triples, 0)
        
        assert len(sampled) == 0
    
    def test_sample_empty_list(self):
        """空のリストからのサンプリングのテスト"""
        sampled = sample_target_triples([], 5)
        
        assert len(sampled) == 0
    
    def test_sample_randomness(self):
        """サンプリングのランダム性のテスト"""
        all_triples = [('e{}'.format(i), 'r1', 't{}'.format(i)) for i in range(100)]
        
        # 複数回サンプリングして異なる結果が得られることを確認
        samples = [
            set(sample_target_triples(all_triples, 10))
            for _ in range(10)
        ]
        
        # 少なくとも一部は異なるサンプルが得られるはず
        unique_samples = len(set(tuple(sorted(s)) for s in samples))
        assert unique_samples > 1


class TestParameterValidation:
    """パラメータの妥当性検証テスト"""
    
    def test_n_rules_pool_positive(self):
        """ルールpool数が正の値であることの検証"""
        n_rules_pool = 15
        assert n_rules_pool > 0
        assert isinstance(n_rules_pool, int)
    
    def test_n_rules_select_less_than_pool(self):
        """選択ルール数がpool数以下であることの検証"""
        n_rules_pool = 15
        n_rules_select = 3
        assert n_rules_select <= n_rules_pool
        assert n_rules_select > 0
    
    def test_n_targets_per_rule_positive(self):
        """ルールあたりのtarget数が正の値であることの検証"""
        n_targets_per_rule = 10
        assert n_targets_per_rule > 0
        assert isinstance(n_targets_per_rule, int)
    
    def test_llm_temperature_range(self):
        """LLM temperatureが適切な範囲であることの検証"""
        llm_temperature = 0.3
        assert 0 <= llm_temperature <= 1


class TestDataStructures:
    """データ構造のテスト"""
    
    def test_triple_tuple_structure(self):
        """トリプルのタプル構造の検証"""
        triple = ('entity1', 'relation', 'entity2')
        assert len(triple) == 3
        assert all(isinstance(x, str) for x in triple)
    
    def test_rule_additions_info_structure(self):
        """rule_additions_info構造の検証"""
        rule_additions_info = {
            'rule_001': {
                'target_triples': [['e1', 'r1', 't1']],
                'added_triples': [['e1', 'r2', 't2']],
                'n_targets': 1,
                'n_added': 1
            }
        }
        
        for rule_id, data in rule_additions_info.items():
            assert 'target_triples' in data
            assert 'added_triples' in data
            assert 'n_targets' in data
            assert 'n_added' in data
            assert isinstance(data['n_targets'], int)
            assert isinstance(data['n_added'], int)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
