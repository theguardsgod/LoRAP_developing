"""Round two: fixed, validation-motivated candidates around each incumbent."""
from copy import deepcopy

CELLS = ['DQ_CiteSeer_GAT', 'QAT_CIFAR10_GCN', 'DQ_ZINC_GCN', 'A2Q_ZINC_GAT']


def candidates(cell, config, hp):
    rows = []

    def add(label, *, lr=None, quant=1., decay=None, scheduler=None, **prompt):
        c, h = deepcopy(config), deepcopy(hp)
        h.pop('prompt_lr_multiplier', None)
        h.update(prompt)
        if lr is not None:
            c['lr'] = lr
        if decay is not None:
            c['weight_decay'] = decay
        if quant != 1.:
            for key in ('lr_feature', 'lr_message', 'lr_weight', 'lr_bit', 'lr_attention', 'lr_edge'):
                c[key] *= quant
        h['lr_scheduler'] = scheduler or dict(name='native')
        rows.append(dict(id=f'r{len(rows):02}', label=label, config=c, hp=h))

    add('previous_selected_configuration')
    cosine = dict(name='cosine')
    slow_step = dict(name='step', step_size=100, gamma=.5)
    plateau = dict(name='plateau', patience=10, factor=.5)
    if cell == 'QAT_CIFAR10_GCN':
        add('lr_002', lr=.002)
        add('lr_004_cosine', lr=.004, scheduler=cosine)
        add('regularized_002_cosine', lr=.002, decay=.0004, scheduler=cosine)
        add('incumbent_cosine', scheduler=cosine)
        add('slow_plateau', lr=.0005, scheduler=plateau)
        add('rank4_fast_prompt', lr=.002, rank=4, prompt_lr=.003, scheduler=cosine)
        add('rank8_cosine', rank=8, lr=.002, scheduler=cosine)
        add('prompt_lr_01', prompt_lr=.01)
        add('weak_input_slow_decay', lr=.002, prompt_lr=.003, input_scale=.1, scheduler=slow_step)
        add('forty_prompts', rank=4, num_prompts=40, input_scale=.1, scheduler=cosine)
        add('small_xavier', zero_init=False, input_scale=.1, aggregation_scale=.1, scheduler=cosine)
        add('no_backbone_decay', lr=.002, decay=0., prompt_lr=.0003, input_scale=.1, scheduler=cosine)
    elif cell == 'DQ_ZINC_GCN':
        add('slow_cosine', lr=.0005, scheduler=cosine)
        add('slower_cosine', lr=.00025, scheduler=cosine)
        add('fast_cosine', lr=.002, scheduler=cosine)
        add('fast_plateau', lr=.003, scheduler=plateau)
        add('incumbent_cosine', scheduler=cosine)
        add('slow_decay', scheduler=slow_step)
        add('small_regularized', lr=.0005, decay=.0004, prompt_weight_decay=.0001, scheduler=cosine)
        add('no_backbone_decay', decay=0., scheduler=cosine)
        add('strong_prompt', prompt_lr=.003, aggregation_scale=2., scheduler=cosine)
        add('rank4_slow_backbone', rank=4, lr=.0005, input_scale=.1, scheduler=cosine)
        add('rank8_prompt', rank=8, prompt_lr=.003, input_scale=.1, scheduler=cosine)
        add('forty_prompts', rank=4, num_prompts=40, lr=.0005, scheduler=plateau)
    elif cell == 'DQ_CiteSeer_GAT':
        add('decay_0004', decay=.0004)
        add('decay_001', decay=.001)
        add('decay_005', decay=.005)
        add('prompt_decay', decay=.0004, prompt_weight_decay=.0001)
        add('strong_prompt_decay', decay=.001, prompt_weight_decay=.001)
        add('lr_002', lr=.002, decay=.0004)
        add('lr_003', lr=.003, decay=.001)
        add('cosine', decay=.0004, scheduler=cosine)
        add('slow_decay', decay=.0004, scheduler=slow_step)
        add('rank4_regularized', rank=4, decay=.001)
        add('forty_prompts', num_prompts=40, decay=.0004)
        add('prompt_lr_01', prompt_lr=.01, decay=.001)
    elif cell == 'A2Q_ZINC_GAT':
        add('quant_lr_x03', quant=.3)
        add('quant_lr_x01', quant=.1)
        add('quant_lr_x003', quant=.03)
        add('slow_backbone_quant', lr=.00025, quant=.1)
        add('faster_backbone_slow_quant', lr=.001, quant=.1)
        add('rank8_slow_quant', rank=8, num_prompts=20, prompt_lr=.001, quant=.1)
        add('xavier_slow_quant', num_prompts=20, zero_init=False, quant=.1)
        add('strong_prompt_slow_quant', num_prompts=20, prompt_lr=.003, quant=.3)
        add('patient_scheduler', quant=.1, scheduler=dict(name='plateau', patience=15, factor=.7))
        add('cosine_slow_quant', quant=.1, scheduler=cosine)
        add('very_slow_quant_regularized', lr=.00025, quant=.01, decay=.0001)
        add('strong_input_slow_quant', input_scale=1., prompt_lr=.001, quant=.03)
    else:
        raise ValueError(cell)
    if len(rows) != 13:
        raise AssertionError('Expected incumbent plus 12 new candidates')
    return rows
