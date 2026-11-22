"""Metadata-only rank/BPW planning for individual Gemma 3 LittleBit projections.

No matrix allocation, QAT, inference, or quality promise. The rank needed
for a target BPW follows the same factor/sign/scale accounting used by
training.layer_train. It is not a recommendation of recovered model accuracy.
"""
from __future__ import annotations

import math

from .batch import scan_source
from .layer_train import estimated_effective_bits


def build_rank_plan(source, *, layer=0, projection="self_attn.q_proj",
                    ranks=(32, 64, 128, 256), target_bpw=0.55,
                    max_teacher_mib=512, allow_test_shapes=False,
                    residual=True):
    """Compute physical bits-per-weight from Safetensors tensor dimensions."""
    if (type(target_bpw) not in (int, float) or not math.isfinite(target_bpw)
            or not 0 < target_bpw <= 16):
        raise ValueError("target_bpw must be a finite positive bits-per-weight target")
    if not isinstance(residual, bool):
        raise ValueError("residual must be boolean")
    if (not isinstance(ranks, (list, tuple)) or not ranks
            or len(ranks) > 20 or len(ranks) != len(set(ranks))
            or any(type(rank) is not int or rank < 1 for rank in ranks)):
        raise ValueError("ranks must be distinct positive integers")
    inventory = scan_source(source, layer_start=layer, layer_end=layer + 1,
                            projections=[projection], max_teacher_mib=max_teacher_mib,
                            allow_test_shapes=allow_test_shapes)
    chosen = inventory["projections"][0]
    rows, cols = chosen["shape"]
    smallest = min(rows, cols)
    if any(rank > smallest for rank in ranks):
        raise ValueError("Selected rank exceeds the dimensions of the original teacher projection")

    records = []
    for rank in ranks:
        bits = estimated_effective_bits(rows, cols, rank, residual)
        records.append({
            "rank": rank, "estimated_effective_bpw": bits,
            "packed_dense_projection_ratio": bits / 16,
            "relative_rank": rank / smallest,
            "quality_known": False,
            "full_model_quality_known": False,
        })

    # Solve the target BPW accounting equation. Residual factors duplicate
    # binary U,V signs and floating magnitude vectors, but no full dense matrix.
    scale_cost = (32 if residual else 16) * (rows + cols)
    factor_cost_per_rank = (2 if residual else 1) * (rows + cols + 16)
    need = math.ceil((target_bpw * rows * cols - scale_cost) / factor_cost_per_rank)
    target_rank = max(1, need)
    meets_target = target_rank <= smallest
    if not meets_target:
        target_rank = smallest
    bpw = estimated_effective_bits(rows, cols, target_rank, residual)
    return {
        "status": "RANK_BUDGET_PLAN_ONLY",
        "source": str(source),
        "layer": layer, "projection": projection,
        "teacher_shape": [rows, cols],
        "teacher_dtype": chosen["dtype"],
        "target_bpw": float(target_bpw),
        "residual_factors": residual,
        "rank_for_target_bpw": target_rank if meets_target else None,
        "target_bpw_reachable_with_this_accounting": meets_target,
        "estimated_bpw_at_target_rank": bpw,
        "options": records,
        "trains_or_allocates_weights": False,
        "trains_full_model": False,
        "predicts_model_quality": False,
        "note": "Bit accounting is exact for the implemented factor/scales "
                "formula, not measured disk size, numerical fidelity or language quality.",
    }
