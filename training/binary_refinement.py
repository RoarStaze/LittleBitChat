"""Greedy binary coordinate-descent refinement at unchanged LittleBit bitwidth.

A selected channel has a signed U column and signed V row. Holding all
other channels fixed, we greedily minimize dense teacher squared error over
those two sign vectors. Original factor magnitudes and scale vector shapes
are unchanged; no additional bits are added to a packed projection.

This is a local training-free initializer, not guaranteed global optimum,
language quality, or activation-aware calibration.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def refine_binary_signs(model, teacher, *, channels=8):
    """Greedily refine up to `channels` high-energy binary rank components.

    Sign choices for each selected U column and then V row are the exact
    row/column minimizers for that fixed counterpart, holding all other
    channels fixed. A channel update is accepted only for lower teacher SSE;
    the entire model is restored if the final FP32 reconstruction regresses.
    """
    if type(channels) is not int or not 0 <= channels <= 256:
        raise ValueError("binary refinement channels must be an integer in 0..256")
    if (not isinstance(teacher, torch.Tensor) or teacher.ndim != 2
            or not teacher.is_floating_point()
            or not bool(torch.isfinite(teacher).all())):
        raise ValueError("Expected a finite floating-point 2D teacher")
    if tuple(teacher.shape) != tuple(model.dense_weight().shape):
        raise ValueError("Teacher shape disagrees with the binary model")
    target = teacher.detach().to(dtype=torch.float32, device=model.u1.device)
    if not bool(torch.isfinite(model.dense_weight()).all()):
        raise ValueError("Binary factors must be finite")
    before_weight = model.dense_weight()
    residual = target - before_weight
    before = float(residual.square().sum())
    if not math.isfinite(before):
        raise ValueError("Nonfinite baseline reconstruction loss")
    if channels == 0:
        return dict(changed=False, examined_channels=0, accepted_channels=0,
                    weight_sse_before=before, weight_sse_after=before,
                    algorithm="binary_coordinate_descent_v1")

    ranked = []
    for suffix in ("", "_R") if model.residual else ("",):
        U, V = getattr(model, "U"+suffix), getattr(model, "V"+suffix)
        u1, u2, v1, v2 = [
            getattr(model, name+suffix) for name in ("u1", "u2", "v1", "v2")
        ]
        if V.shape[0] != U.shape[1]:
            raise ValueError("Invalid binary channel shapes")
        # Estimate each channel's Frobenius contribution with exact scales.
        row_power = float(u1.square().sum())
        col_power = float(v2.square().sum())
        for k in range(U.shape[1]):
            energy = row_power * col_power * float((u2[k]*v1[k]).square())
            if not math.isfinite(energy):
                raise ValueError("Nonfinite binary-channel energy")
            ranked.append((energy, suffix, k))
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    selection = ranked[:channels]
    originals = {}
    accepted = 0

    for _, suffix, k in selection:
        U, V = getattr(model, "U"+suffix), getattr(model, "V"+suffix)
        u1, u2, v1, v2 = [
            getattr(model, name+suffix) for name in ("u1", "u2", "v1", "v2")
        ]
        # Signed outer-product contribution of this one component.
        coeff = u1 * (u2[k] * v1[k])
        old_u = U[:, k].clone()
        old_v = V[k, :].clone()
        a = torch.where(old_u >= 0, coeff, -coeff)
        b = torch.where(old_v >= 0, v2, -v2)
        other_residual = residual + torch.outer(a, b)
        prior_sse = float(residual.square().sum())

        # For fixed V, every U sign minimizes its own target output row.
        ugain = coeff * (other_residual @ b)
        U[:, k].copy_(torch.where(ugain > 0, old_u.abs(),
                         torch.where(ugain < 0, -old_u.abs(), old_u)))
        new_u = U[:, k]
        a = torch.where(new_u >= 0, coeff, -coeff)

        # For fixed U, every V sign minimizes its own input column.
        vgain = v2 * (other_residual.T @ a)
        V[k, :].copy_(torch.where(vgain > 0, old_v.abs(),
                         torch.where(vgain < 0, -old_v.abs(), old_v)))
        new_v = V[k, :]
        b = torch.where(new_v >= 0, v2, -v2)
        candidate_residual = other_residual - torch.outer(a, b)
        candidate_sse = float(candidate_residual.square().sum())

        if math.isfinite(candidate_sse) and candidate_sse < prior_sse:
            if suffix not in originals:
                originals[suffix] = {}
            originals[suffix][k] = (old_u, old_v)
            residual = candidate_residual
            accepted += 1
        else:
            U[:, k].copy_(old_u)
            V[k, :].copy_(old_v)

    # Recompute through the actual packed-equivalent path, not a surrogate.
    after = float((target - model.dense_weight()).square().sum())
    if not math.isfinite(after) or after > before:
        for suffix, entries in originals.items():
            U, V = getattr(model, "U"+suffix), getattr(model, "V"+suffix)
            for k, (original_u, original_v) in entries.items():
                U[:, k].copy_(original_u)
                V[k, :].copy_(original_v)
        after = before
        accepted = 0

    return dict(changed=after < before, examined_channels=len(selection),
                accepted_channels=accepted, weight_sse_before=before,
                weight_sse_after=after,
                algorithm="binary_coordinate_descent_v1")
