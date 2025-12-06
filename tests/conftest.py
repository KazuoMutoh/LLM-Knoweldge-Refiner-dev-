"""
pytest設定とフィクスチャ

共通のテストデータとモックを提供
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json


@pytest.fixture
def temp_dir():
    """一時ディレクトリのフィクスチャ"""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_triples():
    """サンプルトリプルデータ"""
    return [
        ('entity_1', '/location/location/contains', 'entity_2'),
        ('entity_3', '/location/location/contains', 'entity_4'),
        ('entity_5', '/location/location/contains', 'entity_6'),
        ('entity_7', '/location/location/contains', 'entity_8'),
        ('entity_9', '/location/location/contains', 'entity_10'),
    ]


@pytest.fixture
def sample_config_dataset():
    """サンプル設定データ"""
    return {
        "dataset_name": "test_dataset",
        "target_relation": "/location/location/contains",
        "n_train": 100,
        "n_valid": 20,
        "n_test": 20
    }


@pytest.fixture
def sample_config_embedding():
    """埋込モデル設定データ"""
    return {
        "model_name": "TransE",
        "embedding_dim": 50,
        "training_kwargs": {
            "num_epochs": 10,
            "batch_size": 128
        }
    }


@pytest.fixture
def mock_dataset_dir(temp_dir, sample_triples, sample_config_dataset):
    """モックデータセットディレクトリ"""
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()
    
    # トリプルファイルの作成
    train_file = dataset_dir / "train.txt"
    with open(train_file, 'w') as f:
        for h, r, t in sample_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    
    valid_file = dataset_dir / "valid.txt"
    with open(valid_file, 'w') as f:
        f.write("valid_e1\t/location/location/contains\tvalid_t1\n")
    
    test_file = dataset_dir / "test.txt"
    with open(test_file, 'w') as f:
        f.write("test_e1\t/location/location/contains\ttest_t1\n")
    
    target_file = dataset_dir / "target_triples.txt"
    with open(target_file, 'w') as f:
        for triple in sample_triples[:3]:
            f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")
    
    # 設定ファイルの作成
    config_file = dataset_dir / "config_dataset.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config_dataset, f)
    
    return dataset_dir


@pytest.fixture
def sample_rule_pool():
    """サンプルルールpool"""
    from simple_active_refine.amie import AmieRule, TriplePattern
    from simple_active_refine.rule_selector import RuleWithId
    
    rules = []
    for i in range(5):
        head = TriplePattern(
            s=f'?a',
            p='/location/location/contains',
            o=f'?b'
        )
        body = [
            TriplePattern(
                s=f'?a',
                p=f'/relation_{i}',
                o=f'?c'
            ),
            TriplePattern(
                s=f'?c',
                p='/location/location/contains',
                o=f'?b'
            )
        ]
        
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
            metadata={'rule_id': f'rule_{i:03d}'}
        )
        rules.append(RuleWithId(rule_id=f'rule_{i:03d}', rule=rule))
    
    return rules


@pytest.fixture
def sample_added_triples():
    """サンプル追加トリプル"""
    return [
        ('new_entity_1', '/location/location/contains', 'new_entity_2'),
        ('new_entity_3', '/location/location/contains', 'new_entity_4'),
    ]


@pytest.fixture
def sample_rule_additions_info():
    """サンプルルール追加情報"""
    return {
        'rule_001': {
            'target_triples': [
                ['entity_1', '/location/location/contains', 'entity_2']
            ],
            'added_triples': [
                ['new_entity_1', '/location/location/contains', 'new_entity_2']
            ],
            'n_targets': 1,
            'n_added': 1
        }
    }
