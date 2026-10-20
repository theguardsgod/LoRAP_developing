"""Dynamic residual codes; no learned parameters or reference model required."""

import torch
from torch_geometric.utils import remove_self_loops

from ..models import sum_messages


def graph_operator(x, edges, eps):
    """GIN sum, including the separately weighted self term."""
    edges, _ = remove_self_loops(edges)
    return (1 + eps) * x + sum_messages(x, edges)


def encode(residual, basis, scale):
    values = residual @ basis
    normalized = values / scale
    codes = (normalized.abs().add(.5).floor() * normalized.sign()).clamp(-127, 127).to(torch.int8)
    return codes, values


def propagate_codes(codes, edges, eps, scale):
    """INT32 neighbor accumulation; real-valued epsilon is applied afterwards.

    No full-width floating residual is gathered along edges. The conservative
    edge-count bound prevents INT32 overflow even with repeated edges.
    """
    edges, _ = remove_self_loops(edges)
    if edges.size(1) * 127 > torch.iinfo(torch.int32).max:
        raise OverflowError("INT32 accumulation bound exceeded")
    total = torch.zeros(codes.shape, device=codes.device, dtype=torch.int32)
    total.index_add_(0, edges[1], codes[edges[0]].to(torch.int32))
    return (total.to(scale.dtype) + (1 + eps) * codes.to(scale.dtype)) * scale


def shuffle_codes(codes, batch, generator):
    """Permutation within each graph; never exchanges information across graphs."""
    # PyG batches have contiguous graph blocks. A stable random sort within
    # each block keeps this operation on CUDA, without per-graph host syncs.
    key = batch.to(torch.float64) + torch.rand(len(batch), device=batch.device,
                                            generator=generator, dtype=torch.float64)
    return codes[torch.argsort(key, stable=True)]


def residual_correction(residual, edges, eps, basis, scale, batch=None,
                        shuffle=False, generator=None, floating=False):
    codes, values = encode(residual, basis, scale)
    if floating:
        decoded = values
        correction = graph_operator(values, edges, eps) @ basis.T
    else:
        if shuffle:
            codes = shuffle_codes(codes, batch, generator)
        decoded = codes.to(values.dtype) * scale
        correction = propagate_codes(codes, edges, eps, scale) @ basis.T
    return correction, dict(codes=codes, values=values, decoded=decoded)


def higher_precision(x, quantizer):
    """Add one bit, holding each channel's clipping range fixed."""
    bits = quantizer.bit.detach().abs().round().flatten()
    step = quantizer.alpha.detach().abs().flatten()
    qmax = 2 ** (bits - 1) - 1
    new_qmax = 2 ** bits - 1
    new_step = step * qmax / new_qmax
    normalized = (x / new_step).clamp(-new_qmax, new_qmax)
    return normalized.sign() * normalized.abs().add(.5).floor() * new_step


def selected_precision(x, quantizer, indices):
    """One call to the existing quantizer with calibrated mixed-bit metadata."""
    bits = quantizer.bit.detach().abs().round().clone()
    step = quantizer.alpha.detach().abs().clone()
    old_max = 2 ** (bits[indices] - 1) - 1
    bits[indices] += 1
    step[indices] *= old_max / (2 ** (bits[indices] - 1) - 1)
    return quantizer.quant_weight_func.apply(x.T, step, bits).T


def select_channels(gains, limit):
    order = torch.argsort(gains, descending=True, stable=True)
    return order[gains[order] > 0][:limit]


def canonical_basis(error, rank=2):
    """Uncentered second moment, because the objective is reconstruction."""
    gram = error.double().T @ error.double()
    values, vectors = torch.linalg.eigh(gram)
    basis = vectors[:, -rank:].flip(1)
    indices = basis.abs().argmax(0)
    signs = basis[indices, torch.arange(rank, device=error.device)].sign()
    basis = basis * signs
    return basis.float(), values.flip(0).clamp_min(0)


def error_terms(residual, correction, edges, eps, basis=None, decoded=None):
    """Exact orthogonal decomposition, evaluated in double precision."""
    target = graph_operator(residual.double(), edges, eps.double())
    after = target - correction.double()
    before_sq, after_sq = target.square().sum(), after.square().sum()
    terms = dict(before_sq=before_sq, after_sq=after_sq,
                 correction_sq=correction.double().square().sum(),
                 dot=(target * correction.double()).sum())
    if basis is not None:
        b = basis.double()
        projection = target @ b
        truncation = target - projection @ b.T
        eta = decoded.double() - residual.double() @ b
        side_error = graph_operator(eta, edges, eps.double())
        terms.update(truncation_sq=truncation.square().sum(),
                     side_quantization_sq=side_error.square().sum(),
                     captured_sq=projection.square().sum())
        rhs = terms["truncation_sq"] + terms["side_quantization_sq"]
        terms["identity_relative_gap"] = (after_sq - rhs).abs() / before_sq.clamp_min(1e-30)
    return terms
