from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

dir_triples = './data/FB15k-237'
#dir_triples = './processed_data/20250826/fb15k-237_drop_nationality'
dir_model = './data/FB15k-237'
# train, test, validをpykeenのTriples Factoryとして読み込む
train_tf = TriplesFactory.from_path(f'{dir_triples}/train.txt')
valid_tf = TriplesFactory.from_path(f'{dir_triples}/dev.tsv')  # Use dev.tsv for validation
test_tf = TriplesFactory.from_path(f'{dir_triples}/test.txt')

result = pipeline(
    # --- Dataset ---
    training=train_tf,
    validation=valid_tf,  # Enable validation for early stopping
    testing=test_tf,

    dataset_kwargs=dict(create_inverse_triples=True),

    # --- Model & loss ---
    model='TransE',
    model_kwargs=dict(
        embedding_dim=64,
        scoring_fct_norm=1,
    ),
    loss='CrossEntropyLoss',

    # --- Training loop & stopper ---
    training_loop='lcwa',
    stopper='early',
    stopper_kwargs=dict(
        frequency=5,
        patience=2,
    ),

    # --- Optimizer ---
    optimizer='adam',
    optimizer_kwargs=dict(
        lr=0.0016608460884079603,
        weight_decay=0.0,
    ),

    # --- Trainer kwargs ---
    training_kwargs=dict(
        num_epochs=10,
        batch_size=256,
        label_smoothing=0.717650072390557,
    ),

    # --- Evaluation ---
    evaluator='RankBasedEvaluator',
    #evaluation_kwargs=dict(filtered=True),

    # 再現性（任意）
    random_seed=42,
)
print(result.metric_results.to_dict())  # 指標一式（Hits@K / MRRなど）
result.save_to_directory("./models/20250903/pykeen_test_transe")  # モデル保存
