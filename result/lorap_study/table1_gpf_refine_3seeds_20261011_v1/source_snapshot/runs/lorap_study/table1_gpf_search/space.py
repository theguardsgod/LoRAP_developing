"""Fixed search space registered before new model evaluations."""
from ..table1_gpf_rerun.run import HP

CELLS = ['DQ_Cora_GIN', 'QAT_CIFAR10_GCN', 'DQ_ZINC_GCN',
         'DQ_CiteSeer_GAT', 'DQ_MNIST_GAT', 'A2Q_ZINC_GAT']
SEEDS = [41, 42, 43]


def candidates():
    # Backbone learning-rate multipliers do not affect its initialization.
    rows = [
        ('gentle_zero', 2, 20, .1, .1, .1, True, 1., 1.),
        ('slow_zero', 2, 20, .1, 1., 1., True, 1., 1.),
        ('zero', 2, 20, 1., 1., 1., True, 1., 1.),
        ('gentle_xavier_r4', 4, 20, .1, .1, .1, False, 1., 1.),
        ('zero_r4', 4, 20, 1., 1., 1., True, 1., 1.),
        ('fast_r8', 8, 20, 3., .1, 1., True, 1., 1.),
        ('slow_backbone', 4, 20, 1., .1, 1., True, .5, .3),
        ('alternative_backbone', 4, 10, .3, .1, 1., True, 2., .1),
    ]
    return [dict(id=f'c{i:02}', label=label,
                 hp=dict(HP, rank=r, num_prompts=p, prompt_lr_multiplier=lr,
                         input_scale=ins, aggregation_scale=agg, zero_init=zero),
                 overrides=dict(backbone_lr_multiplier=base, a2q_quantizer_lr_multiplier=quant))
            for i, (label, r, p, lr, ins, agg, zero, base, quant) in enumerate(rows)]


def instantiate(candidate, config):
    c, hp = dict(config), dict(candidate['hp'])
    base = candidate['overrides']['backbone_lr_multiplier']
    if c['framework'] == 'A2Q' and candidate['id'] == 'c07':
        base = .5
    c['lr'] *= base
    if c['framework'] == 'A2Q':
        multiplier = candidate['overrides']['a2q_quantizer_lr_multiplier']
        for key in ('lr_feature', 'lr_message', 'lr_weight', 'lr_bit', 'lr_attention', 'lr_edge'):
            c[key] *= multiplier
    hp['prompt_lr'] = config['lr'] * hp['prompt_lr_multiplier']
    return c, hp
