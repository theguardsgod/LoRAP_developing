"""Expanded training search for QAT CIFAR10 GCN; fixed architecture and quantizers."""
from copy import deepcopy

CELLS = ['QAT_CIFAR10_GCN']


def candidates(cell, config, hp):
    if cell not in CELLS:
        raise ValueError(cell)
    rows = [dict(id='r00', label='previous_selected_configuration',
                 config=deepcopy(config), hp=deepcopy(hp))]

    def add(label, *, lr=None, decay=None, dropout=None, bn=None, batch=None,
            scheduler=None, **prompt):
        c, h = deepcopy(config), deepcopy(hp)
        for key, value in [('lr', lr), ('weight_decay', decay), ('head_dropout', dropout),
                           ('batchnorm_momentum', bn), ('batch_size', batch)]:
            if value is not None:
                c[key] = value
        h.update(prompt)
        if scheduler is not None:
            h['lr_scheduler'] = deepcopy(scheduler)
        rows.append(dict(id=f'r{len(rows):02}', label=label, config=c, hp=h, epochs=300))

    cosine = dict(name='cosine')
    add('incumbent_300_epochs')
    add('head_dropout_zero', dropout=0.)
    add('head_dropout_01', dropout=.1)
    add('head_dropout_025', dropout=.25)
    add('dropout_01_slow_bn', dropout=.1, bn=.01)
    add('dropout_01_medium_bn', dropout=.1, bn=.03)
    add('dropout_01_batch64', dropout=.1, batch=64)
    add('dropout_01_batch256', dropout=.1, batch=256)
    add('dropout_zero_slow_bn', dropout=0., bn=.01)
    add('adamw_decay_001', dropout=.1, optimizer='adamw', decay=.001)
    add('adamw_decay_01', dropout=.1, optimizer='adamw', decay=.01)
    add('adamw_fast_cosine', dropout=.1, optimizer='adamw', decay=.001,
        lr=.001, scheduler=cosine)
    add('fast_backbone_and_prompt', dropout=.1, lr=.001, prompt_lr=.01, scheduler=cosine)
    add('slow_backbone_reduced_dropout', dropout=.1, lr=.0001)
    add('rank8_fast_prompt', dropout=.25, lr=.0005, rank=8, num_prompts=40,
        prompt_lr=.01, scheduler=cosine)
    add('rank16_eighty_prompts', dropout=.1, rank=16, num_prompts=80, scheduler=cosine)
    add('rank8_no_dropout_weak_input', dropout=0., rank=8, num_prompts=40,
        prompt_lr=.001, input_scale=.1, scheduler=cosine)
    add('weak_slow_prompt', dropout=.1, prompt_lr=.0001, input_scale=.1, aggregation_scale=.1)
    add('very_fast_prompt', dropout=.1, prompt_lr=.03, input_scale=.1)
    add('rank8_xavier', dropout=.1, rank=8, num_prompts=40, zero_init=False,
        input_scale=.1, aggregation_scale=.1, lr=.0005)
    add('strong_l2_regularization', dropout=.1, lr=.0005, decay=.001, prompt_weight_decay=.001)
    add('adamw_batch64', dropout=.1, batch=64, optimizer='adamw', lr=.0005,
        decay=.01, scheduler=cosine)
    add('adamw_batch256', dropout=.1, batch=256, optimizer='adamw', lr=.001,
        decay=.01, scheduler=cosine)
    add('rank8_slow_bn', dropout=.25, bn=.01, rank=8, num_prompts=40,
        lr=.0005, scheduler=cosine)
    if len(rows) != 25:
        raise AssertionError('Expected incumbent plus 24 new candidates')
    return rows
