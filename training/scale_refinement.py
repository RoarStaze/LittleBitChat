"""Training-free nonnegative output-row calibration for fixed signed factor paths.

Fits a joint 2-column least squares model per teacher output row, with
nonnegative row gains. Does not change factor signs, ranks, export tensor
structure, or effective bits-per-weight. It is not language-quality evidence.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def refine_output_row_scales(model, teacher):
    """Refit output magnitude vectors u1 and optional u1_R using dense weights.

    Considers the interior nonnegative 2x2 least-squares solution, both
    nonnegative boundary solutions, zero and the untouched baseline per row.
    Never chooses coefficients from diagnostic holdout probes.
    """
    if (not isinstance(teacher, torch.Tensor) or teacher.ndim != 2
            or not teacher.is_floating_point() or not bool(torch.isfinite(teacher).all())):
        raise ValueError("Expected a finite 2D floating-point teacher")
    target = teacher.detach().to(device=model.u1.device, dtype=torch.float32)
    if tuple(target.shape) != tuple(model.dense_weight().shape):
        raise ValueError("Teacher dimensions do not match signed factors")

    primary = model.dense_path_weight("")
    residual = (model.dense_path_weight("_R") if model.residual
                else torch.zeros_like(primary))
    before = (primary + residual - target).square().sum().item()
    if not math.isfinite(before):
        raise ValueError("Nonfinite initial reconstruction loss")

    # Accumulate in FP64 for stable 2x2 normal equations at large ranks.
    aa = (primary * primary).sum(dim=1, dtype=torch.float64)
    bb = (residual * residual).sum(dim=1, dtype=torch.float64)
    ab = (primary * residual).sum(dim=1, dtype=torch.float64)
    ya = (primary * target).sum(dim=1, dtype=torch.float64)
    yb = (residual * target).sum(dim=1, dtype=torch.float64)
    one = torch.ones_like(aa)
    zero = torch.zeros_like(aa)
    best_a = one.clone()
    best_b = one.clone() if model.residual else zero.clone()

    def score(a, b):
        # ||teacher||^2 is constant over all candidate row gains.
        return aa*a*a + 2*ab*a*b + bb*b*b - 2*ya*a - 2*yb*b

    best_cost = score(best_a, best_b)

    def consider(a, b, allowed):
        nonlocal best_a, best_b, best_cost
        cost = score(a, b)
        take = (allowed & torch.isfinite(a) & torch.isfinite(b)
                & torch.isfinite(cost) & (cost < best_cost))
        best_a = torch.where(take, a, best_a)
        best_b = torch.where(take, b, best_b)
        best_cost = torch.where(take, cost, best_cost)

    consider(zero, zero, torch.ones_like(aa, dtype=torch.bool))
    safe_aa = torch.where(aa > 1e-30, aa, one)
    consider((ya / safe_aa).clamp_min(0), zero, aa > 1e-30)
    if model.residual:
        safe_bb = torch.where(bb > 1e-30, bb, one)
        consider(zero, (yb / safe_bb).clamp_min(0), bb > 1e-30)
        determinant = aa * bb - ab.square()
        valid = determinant > (aa * bb).clamp_min(1e-30) * 1e-10
        safe_det = torch.where(valid, determinant, one)
        a_joint = (ya * bb - yb * ab) / safe_det
        b_joint = (yb * aa - ya * ab) / safe_det
        consider(a_joint, b_joint, valid & (a_joint >= 0) & (b_joint >= 0))

    previous_a = model.u1.detach().clone()
    previous_b = model.u1_R.detach().clone() if model.residual else None
    next_a = previous_a * best_a.to(previous_a.dtype)
    next_b = previous_b * best_b.to(previous_b.dtype) if model.residual else None
    if not bool(torch.isfinite(next_a).all()) or (
            next_b is not None and not bool(torch.isfinite(next_b).all())):
        raise ValueError("Nonfinite refined output scales")

    model.u1.copy_(next_a)
    if model.residual:
        model.u1_R.copy_(next_b)
    after = (model.dense_weight() - target).square().sum().item()
    # Reconstruct with the actual FP32 factors and reject numerical regressions.
    if not math.isfinite(after) or after > before * (1 + 1e-7) + 1e-9:
        model.u1.copy_(previous_a)
        if model.residual:
            model.u1_R.copy_(previous_b)
        return {"changed": False, "weight_sse_before": before,
                "weight_sse_after": before, "regressed": False}
    return {"changed": bool(after < before), "weight_sse_before": before,
            "weight_sse_after": after, "regressed": False}
