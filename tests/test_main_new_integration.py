"""
統合テスト: main.py

テスト戦略:
1. 初期化フェーズのテスト (Phase 0)
   - ディレクトリ構造の作成
   - 初期データのコピー
   - 設定ファイルの読み込み
   
2. 反復フェーズの1サイクルテスト (Phase 1)
   - モックを使用して外部依存を排除
   - データフローの整合性検証
   - ファイル生成の検証
   
3. エンドツーエンドテスト（簡易版）
   - 小規模データセットでの実行
   - 全体的なデータフローの検証
"""

import pytest
import os
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys

sys.path.insert(0, '/app')


class TestInitializationPhase:
    """初期化フェーズのテスト"""
    
    @pytest.fixture
    def mock_initial_data_dir(self, tmp_path):
        """初期データディレクトリのモック作成"""
        initial_dir = tmp_path / "initial_data"
        initial_dir.mkdir()
        
        # 必要なファイルを作成
        (initial_dir / "train.txt").write_text("e1\tr1\tt1\n")
        (initial_dir / "valid.txt").write_text("e2\tr1\tt2\n")
        (initial_dir / "test.txt").write_text("e3\tr1\tt3\n")
        (initial_dir / "target_triples.txt").write_text("e4\tr1\tt4\ne5\tr1\tt5\n")
        
        config = {
            "target_relation": "/location/location/contains",
            "dataset_name": "test_dataset"
        }
        (initial_dir / "config_dataset.json").write_text(json.dumps(config))
        
        return initial_dir
    
    @pytest.fixture
    def mock_working_dir(self, tmp_path):
        """作業ディレクトリのモック作成"""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        return working_dir
    
    def test_iter_0_directory_creation(self, mock_initial_data_dir, mock_working_dir):
        """Iteration 0のディレクトリ作成テスト"""
        dir_iter_0 = mock_working_dir / "iter_0"
        
        # 初期データをコピー
        shutil.copytree(mock_initial_data_dir, dir_iter_0)
        
        assert dir_iter_0.exists()
        assert (dir_iter_0 / "train.txt").exists()
        assert (dir_iter_0 / "valid.txt").exists()
        assert (dir_iter_0 / "test.txt").exists()
        assert (dir_iter_0 / "config_dataset.json").exists()
    
    def test_config_loading(self, mock_initial_data_dir):
        """設定ファイルの読み込みテスト"""
        config_path = mock_initial_data_dir / "config_dataset.json"
        
        with open(config_path, 'r') as fin:
            config = json.load(fin)
        
        assert 'target_relation' in config
        assert 'dataset_name' in config
        assert config['target_relation'] == "/location/location/contains"
    
    def test_target_triples_loading(self, mock_initial_data_dir):
        """target triplesの読み込みテスト"""
        target_file = mock_initial_data_dir / "target_triples.txt"
        
        all_target_triples = []
        with open(target_file, 'r') as fin:
            for row in fin:
                words = row.rstrip().split('\t')
                all_target_triples.append((words[0], words[1], words[2]))
        
        assert len(all_target_triples) == 2
        assert ('e4', 'r1', 't4') in all_target_triples
        assert ('e5', 'r1', 't5') in all_target_triples


class TestIterationPhase:
    """反復フェーズのテスト"""
    
    @pytest.fixture
    def mock_rule_with_id(self):
        """RuleWithIdのモック"""
        from simple_active_refine.rule_selector import RuleWithId
        from simple_active_refine.amie import AmieRule, TriplePattern
        
        # 簡単なルールを作成
        head = TriplePattern(s='?a', p='/location/location/contains', o='?b')
        body = [TriplePattern(s='?a', p='/location/location/nearby', o='?c'),
                TriplePattern(s='?c', p='/location/location/contains', o='?b')]
        
        rule = AmieRule(
            head=head, 
            body=body, 
            support=None,
            std_conf=None,
            pca_conf=None,
            head_coverage=None,
            body_size=None,
            pca_body_size=None,
            raw="",
            metadata={'rule_id': 'rule_001'}
        )
        return RuleWithId(rule_id='rule_001', rule=rule)
    
    def test_rule_additions_structure(self, mock_rule_with_id):
        """ルール追加情報の構造テスト"""
        added_triples_by_rule = {
            'rule_001': {
                'rule': mock_rule_with_id.rule,
                'target_triples': [('e1', 'r1', 't1'), ('e2', 'r1', 't2')],
                'added_triples': [('e1', 'r2', 't3')],
                'details': {}
            }
        }
        
        assert 'rule_001' in added_triples_by_rule
        data = added_triples_by_rule['rule_001']
        assert 'rule' in data
        assert 'target_triples' in data
        assert 'added_triples' in data
        assert len(data['target_triples']) == 2
        assert len(data['added_triples']) == 1
    
    def test_triple_merging(self):
        """トリプルのマージテスト"""
        # 元のトリプルセット
        original_triples = {
            ('e1', 'r1', 't1'),
            ('e2', 'r1', 't2'),
        }
        
        # 追加トリプル
        added_triples = {
            ('e3', 'r1', 't3'),
            ('e4', 'r2', 't4'),
        }
        
        # マージ
        merged = original_triples | added_triples
        
        assert len(merged) == 4
        assert ('e1', 'r1', 't1') in merged
        assert ('e3', 'r1', 't3') in merged
    
    def test_duplicate_triple_handling(self):
        """重複トリプルの処理テスト"""
        original_triples = {
            ('e1', 'r1', 't1'),
            ('e2', 'r1', 't2'),
        }
        
        # 重複を含む追加トリプル
        added_triples = {
            ('e1', 'r1', 't1'),  # 重複
            ('e3', 'r1', 't3'),
        }
        
        merged = original_triples | added_triples
        
        # 重複は除去される
        assert len(merged) == 3
        assert merged == {('e1', 'r1', 't1'), ('e2', 'r1', 't2'), ('e3', 'r1', 't3')}
    
    def test_rule_additions_json_serialization(self, tmp_path):
        """ルール追加情報のJSON保存テスト"""
        rule_additions_info = {
            'rule_001': {
                'target_triples': [['e1', 'r1', 't1']],
                'added_triples': [['e1', 'r2', 't2']],
                'n_targets': 1,
                'n_added': 1
            }
        }
        
        output_file = tmp_path / "rule_additions.json"
        with open(output_file, 'w') as fout:
            json.dump(rule_additions_info, fout, indent=2)
        
        assert output_file.exists()
        
        # 読み込んで検証
        with open(output_file, 'r') as fin:
            loaded = json.load(fin)
        
        assert loaded == rule_additions_info


class TestFileOperations:
    """ファイル操作のテスト"""
    
    def test_file_copying_between_iterations(self, tmp_path):
        """イテレーション間のファイルコピーテスト"""
        # 元ディレクトリ
        dir_current = tmp_path / "iter_0"
        dir_current.mkdir()
        
        files_to_copy = ['test.txt', 'valid.txt', 'train_removed.txt',
                        'test_removed.txt', 'valid_removed.txt',
                        'config_dataset.json', 'target_triples.txt']
        
        for filename in files_to_copy:
            (dir_current / filename).write_text(f"content of {filename}")
        
        # コピー先ディレクトリ
        dir_next = tmp_path / "iter_1"
        dir_next.mkdir()
        
        # ファイルコピー
        for filename in files_to_copy:
            f_src = dir_current / filename
            f_dst = dir_next / filename
            if f_src.exists():
                shutil.copy(f_src, f_dst)
        
        # 検証
        for filename in files_to_copy:
            assert (dir_next / filename).exists()
            assert (dir_next / filename).read_text() == f"content of {filename}"
    
    def test_train_file_update(self, tmp_path):
        """trainファイルの更新テスト"""
        dir_iter = tmp_path / "iter_1"
        dir_iter.mkdir()
        
        # 更新されたトリプルセット
        updated_triples = {
            ('e1', 'r1', 't1'),
            ('e2', 'r1', 't2'),
            ('e3', 'r2', 't3'),
        }
        
        # ファイルに保存
        train_file = dir_iter / 'train.txt'
        with open(train_file, 'w') as fout:
            for h, r, t in updated_triples:
                fout.write(f'{h}\t{r}\t{t}\n')
        
        # 読み込んで検証
        loaded_triples = []
        with open(train_file, 'r') as fin:
            for line in fin:
                parts = line.strip().split('\t')
                loaded_triples.append(tuple(parts))
        
        assert set(loaded_triples) == updated_triples


class TestEdgeCases:
    """エッジケースのテスト"""
    
    def test_empty_target_triples_list(self):
        """空のtarget triplesリストの処理"""
        from main import sample_target_triples
        
        result = sample_target_triples([], 10)
        assert result == []
    
    def test_single_target_triple(self):
        """単一のtarget tripleの処理"""
        from main import sample_target_triples
        
        triples = [('e1', 'r1', 't1')]
        result = sample_target_triples(triples, 5)
        
        assert len(result) == 1
        assert result[0] == ('e1', 'r1', 't1')
    
    def test_no_added_triples(self):
        """トリプルが追加されなかった場合の処理"""
        original_triples = {('e1', 'r1', 't1'), ('e2', 'r1', 't2')}
        added_triples = set()  # 空
        
        merged = original_triples | added_triples
        
        assert merged == original_triples
        assert len(merged) == 2


class TestDataConsistency:
    """データ整合性のテスト"""
    
    def test_target_triple_uniqueness(self):
        """使用済みtarget tripleの重複防止テスト"""
        from main import sample_target_triples
        
        all_triples = [('e{}'.format(i), 'r1', 't{}'.format(i)) for i in range(20)]
        used_triples = set()
        
        # 複数回サンプリング
        for _ in range(3):
            sampled = sample_target_triples(all_triples, 5, exclude_triples=used_triples)
            
            # 重複チェック
            assert len(set(sampled) & used_triples) == 0
            
            used_triples.update(sampled)
        
        # 合計15個が使用されている
        assert len(used_triples) == 15
    
    def test_triple_format_consistency(self):
        """トリプル形式の一貫性テスト"""
        triple = ('entity1', 'relation', 'entity2')
        
        # タプル形式
        assert isinstance(triple, tuple)
        assert len(triple) == 3
        
        # リスト形式への変換
        triple_list = list(triple)
        assert isinstance(triple_list, list)
        assert len(triple_list) == 3
        
        # 相互変換の一貫性
        assert tuple(triple_list) == triple


class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def test_missing_config_file(self, tmp_path):
        """設定ファイルが存在しない場合のエラー"""
        config_path = tmp_path / "nonexistent_config.json"
        
        with pytest.raises(FileNotFoundError):
            with open(config_path, 'r') as fin:
                json.load(fin)
    
    def test_invalid_json_config(self, tmp_path):
        """不正なJSON形式のエラー"""
        config_path = tmp_path / "invalid_config.json"
        config_path.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            with open(config_path, 'r') as fin:
                json.load(fin)
    
    def test_invalid_triple_format(self):
        """不正なトリプル形式の検出"""
        invalid_triple = ('entity1', 'relation')  # 2要素のみ
        
        # 3要素であることを期待
        assert len(invalid_triple) != 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
