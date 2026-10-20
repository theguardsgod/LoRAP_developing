"""Validation-only evolutionary subnet search with resumable candidate caching.

Algorithm follows official GLoRA evolution_utils.py: random population, global
elite parents, per-support mutation, and per-support crossover. Extensions:
validation CE selection, bounded proposal retries, exact RNG resume, and two
initial anchors (all low-rank paths and all disabled) within the same budget.
"""
import math
import random
from pathlib import Path

import torch

from .full_run import atomic_save
from .glora import ARCHITECTURES, CHOICES, NONE_ARCH, REFERENCE_ARCH


def evolutionary_search(evaluate, parameter_count, depth, seed, state_path, *,
                        population=50, generations=20, elite=10, mutation_probability=.2,
                        max_parameters=None, split='val', objective_id=''):
    if split != 'val':
        raise ValueError('GLoRA structure search may only use validation data')
    if min(depth, population, generations, elite) < 1 or elite > population:
        raise ValueError('Invalid evolutionary search dimensions')
    if not 0 <= mutation_probability <= 1 or (max_parameters is not None and max_parameters < 0):
        raise ValueError('Invalid mutation probability or parameter budget')
    path = Path(state_path)
    protocol = dict(depth=depth, seed=seed, population=population, generations=generations, elite=elite,
                    mutation_probability=mutation_probability, max_parameters=max_parameters,
                    split=split, objective_id=objective_id, version=2)
    rng = random.Random(seed + 920_000)
    if path.exists():
        state = torch.load(path, weights_only=False, map_location='cpu')
        if state['protocol'] != protocol:
            raise ValueError('Search protocol or checkpoint changed on resume')
        rng.setstate(state['rng'])
    else:
        state = dict(protocol=protocol, generation=0, population=[], parents=[], records=[],
                     initial_attempted=0, attempts=0, rng=rng.getstate(), complete=False)
    visited = {tuple(record['architecture']) for record in state['records']}

    def save():
        state['rng'] = rng.getstate()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(state, path)

    def random_candidate():
        return tuple(rng.randrange(len(ARCHITECTURES)) for _ in range(depth))

    def mutate(parent):
        return tuple(ARCHITECTURES.index(tuple(rng.choice(options) if rng.random()<mutation_probability else old
                     for old,options in zip(ARCHITECTURES[index],CHOICES))) for index in parent)

    def cross(left, right):
        return tuple(ARCHITECTURES.index(tuple(rng.choice((a,b)) for a,b in
                     zip(ARCHITECTURES[i],ARCHITECTURES[j]))) for i,j in zip(left,right))

    while state['generation'] < generations:
        while len(state['population']) < population:
            state['attempts'] += 1
            attempts = state['attempts']
            if attempts > max(1000, population*200):
                raise RuntimeError('Cannot fill the population with distinct feasible subnets; loosen the parameter budget or reduce population')
            if state['generation'] == 0:
                if state['initial_attempted'] < 2:
                    anchor = (REFERENCE_ARCH, NONE_ARCH)[state['initial_attempted']]
                    state['initial_attempted'] += 1
                    candidate = (anchor,)*depth
                else:
                    candidate = random_candidate()
            else:
                parents = state['parents']
                if attempts >= population*20:
                    candidate = random_candidate()
                elif len(state['population']) < population//2:
                    candidate = mutate(rng.choice(parents))
                else:
                    candidate = cross(rng.choice(parents),rng.choice(parents))
            if candidate in visited:
                continue
            parameters = parameter_count(candidate)
            if max_parameters is not None and parameters > max_parameters:
                continue
            metrics = evaluate(candidate)
            if not math.isfinite(metrics['loss']):
                raise FloatingPointError('Non-finite subnet validation loss')
            state['records'].append(dict(architecture=list(candidate), parameters=parameters,
                                         validation=metrics, generation=state['generation'], order=len(state['records'])))
            state['population'].append(candidate)
            visited.add(candidate)
            save()
        ordered = sorted(state['records'],key=lambda record:(record['validation']['loss'],record['order']))
        state['parents'] = [tuple(record['architecture']) for record in ordered[:elite]]
        state['population'] = []
        state['attempts'] = 0
        state['generation'] += 1
        state['complete'] = state['generation'] == generations
        save()
    best = min(state['records'],key=lambda record:(record['validation']['loss'],record['order']))
    return dict(best=best, evaluated=len(state['records']), protocol=protocol, complete=state['complete'])
