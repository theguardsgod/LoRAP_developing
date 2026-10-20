"""Round three: lower-rate neighborhoods for the two remaining graph tasks."""
from copy import deepcopy

CELLS = ['QAT_CIFAR10_GCN', 'DQ_ZINC_GCN']


def candidates(cell, config, hp):
    # Preserve the incumbent scheduler as well as its model/optimizer settings.
    rows = [dict(id='r00', label='previous_selected_configuration',
                 config=deepcopy(config), hp=deepcopy(hp))]

    def add(label, *, lr=None, decay=None, scheduler=None, **prompt):
        c, h = deepcopy(config), deepcopy(hp)
        h.update(prompt)
        if lr is not None:
            c['lr'] = lr
        if decay is not None:
            c['weight_decay'] = decay
        if scheduler is not None:
            h['lr_scheduler'] = deepcopy(scheduler)
        rows.append(dict(id=f'r{len(rows):02}', label=label, config=c, hp=h))

    if cell == 'QAT_CIFAR10_GCN':
        add('lower_backbone_lr', lr=.00025)
        add('intermediate_backbone_lr', lr=.00075)
        add('patient_plateau', scheduler=dict(name='plateau', patience=20, factor=.5))
        add('low_lr_cosine', scheduler=dict(name='cosine'))
        add('slow_prompt', prompt_lr=.0003)
        add('faster_prompt', prompt_lr=.003)
        add('rank4_low_lr', rank=4)
        add('rank8_low_lr', rank=8)
        add('forty_prompts_low_lr', rank=4, num_prompts=40)
        add('weak_input_low_lr', input_scale=.1)
        add('regularized_low_lr', decay=.0001, prompt_weight_decay=.0001)
        add('slow_backbone_fast_prompt', lr=.00025, prompt_lr=.003)
    elif cell == 'DQ_ZINC_GCN':
        add('lower_backbone_lr', lr=.0001)
        add('intermediate_backbone_lr', lr=.00015)
        add('higher_backbone_lr', lr=.0004)
        add('slow_prompt', prompt_lr=.0003)
        add('faster_prompt', prompt_lr=.003)
        add('rank4_low_lr', rank=4, input_scale=.1)
        add('rank8_low_lr', rank=8, input_scale=.1)
        add('forty_prompts_low_lr', rank=4, num_prompts=40)
        add('no_backbone_decay', decay=0.)
        add('regularized_low_lr', decay=.0001, prompt_weight_decay=.0001)
        add('plateau_low_lr', scheduler=dict(name='plateau', patience=10, factor=.5))
        add('weak_aggregation_low_lr', aggregation_scale=.1)
    else:
        raise ValueError(cell)
    if len(rows) != 13:
        raise AssertionError('Expected incumbent plus 12 new candidates')
    return rows
