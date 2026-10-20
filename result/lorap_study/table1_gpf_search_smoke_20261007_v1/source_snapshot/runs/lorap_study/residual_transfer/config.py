from pathlib import Path

REPO=Path('/home/cyliu/code/A-2Q')
ROOT=REPO/'result/lorap_study/residual_transfer_20260914_v2'
SOURCE=REPO/'result/lorap_study/full_gpu_20260909_v1'
DATASETS=('Cora','CiteSeer','PubMed')
MODELS=('GIN','GCN','GAT')
SEEDS=(41,42,43)
MODES=('none','unit','gate','higher')


def configuration(dataset,architecture):
    c=dict(dataset=dataset,architecture=architecture,bit=4,hidden=16,heads=1,
        baseline_epochs=200,gate_epochs=100,gate_lr=.01,rank=2,side_bits=8,
        lr=.01,weight_decay=.01,dropout=.5,attention_dropout=0.,
        lr_feature=.05,lr_weight=.005,lr_message=.005,lr_bit=.02,
        lr_attention=.005,a_loss=.5,a_storage=5.,split='public',
        data_root=str(REPO/'data/lorap_study') if dataset=='PubMed' else '/data/cyliu/dataset/'+dataset,
        selection='minimum validation cross-entropy; earliest ties; gate epoch zero included',
        higher_max_bits=16,calibration_max_nodes=4096,trainable_quantizer_bit_bounds=[1.,16.])
    if architecture=='GIN':
        c['a_loss']={'Cora':2.,'CiteSeer':.5,'PubMed':.1}[dataset]
        c['recipe_note']='Existing GIN recipe; PubMed uses the CiteSeer GIN recipe with a_loss=0.1 (no dedicated README GIN-PubMed recipe).'
    elif architecture=='GCN':
        c.update(lr_feature=.04,lr_message=.008,lr_bit=.1)
        if dataset=='Cora':
            c.update(lr_weight=.02,a_loss=2.5,dropout=.35,weight_decay=.02)
        elif dataset=='CiteSeer':
            c.update(lr_weight=.008,a_loss=1.5,weight_decay=.015)
        else:
            c.update(lr_feature=.005,lr_weight=.005,lr_message=.005,lr_bit=.02,a_loss=.1)
        c['recipe_note']='README node GCN recipe; raw Planetoid features, no added input prompt.'
    elif architecture=='GAT':
        c.update(hidden=8,heads=8,weight_decay=.001,dropout=.6,attention_dropout=.6,
                 lr_bit=.1,lr_feature=.05,lr_message=.05,lr_attention=.005,a_storage=10.)
        if dataset=='Cora':
            c.update(lr=.01,lr_weight=.005,a_loss=.3)
        elif dataset=='CiteSeer':
            c.update(lr=.005,lr_weight=.01,a_loss=.25)
        else:
            c.update(lr=.005,lr_feature=.005,lr_weight=.001,lr_message=.005,lr_bit=.015,
                     a_loss=.025,attention_dropout=.3)
        c['recipe_note']='README node GAT recipe; preserve quantized attention logits and node-wise value quantizer; use raw logits with CE and omit the historical unused weight parameter.'
    else:
        raise ValueError(architecture)
    return c
