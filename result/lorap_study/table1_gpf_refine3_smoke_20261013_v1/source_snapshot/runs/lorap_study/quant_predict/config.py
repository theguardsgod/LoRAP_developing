from pathlib import Path

REPO = Path('/home/cyliu/code/A-2Q')
ROOT = REPO/'result/lorap_study/quant_predict_20260914_v1'
SOURCE = REPO/'result/lorap_study/residual_transfer_20260914_v2'
DATASETS = ('Cora', 'CiteSeer', 'PubMed')
MODELS = ('GIN', 'GCN', 'GAT')
SEEDS = (41, 42, 43)
VARIANTS = ('task', 'error', 'mlp')


def configuration(dataset, architecture, smoke=False):
    return dict(dataset=dataset, architecture=architecture, variants=list(VARIANTS),
        epochs=2 if smoke else 100, rank=2, num_prompts=20,
        learning_rate=.005 if architecture=='GAT' and dataset!='Cora' else .01,
        weight_decay=0., error_loss_weight=.1, normalization_floor=1e-6,
        context_clip=5., max_diagnostic_nodes=4096, diagnostic_permutations=20,
        frozen_backbone=True, backbone_eval=True, selection='minimum validation CE; epoch 0 included; earliest ties',
        source=str(SOURCE/dataset/architecture), smoke=smoke)
