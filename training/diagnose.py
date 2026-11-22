"""Read-only diagnostics separating truncated SVD error from signed-factor error.

This tool does NOT train, change saved factors, assemble a decoder or measure
language-model quality. The SVD baseline uses randomized torch.svd_lowrank,
not an exact singular-value decomposition or a proven optimum.
"""
from __future__ import annotations

import math

import torch

from .layer_train import (
    LittleBitQAT, TrainingError, _teacher_sha, estimated_effective_bits,
    initialize, load_projection,
)


class DiagnosisError(TrainingError):
    """Unsafe or invalid diagnostic input."""


def _relative_rmse(actual, reference):
    signal = torch.sqrt(torch.mean(reference.square())).item()
    error = torch.sqrt(torch.mean((actual - reference).square())).item()
    return float(error / max(signal, 1e-12))


@torch.no_grad()
def diagnose_matrix(weight, *, rank, seed=42, itq_iterations=20, residual=True,
                    probe_count=32):
    """Estimate errors for a single dense teacher matrix and binary factors.

    Both matrix-relative and random-probe metrics are disclosed. Independent
    untrained teacher-weight reproduction, NOT text generation or whole LLM QAT.
    """
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise DiagnosisError("Expected one finite 2D teacher matrix")
    if not weight.is_floating_point() or not bool(torch.isfinite(weight).all()):
        raise DiagnosisError("Expected finite floating-point teacher weights")
    if type(rank) is not int or not 1 <= rank <= min(weight.shape):
        raise DiagnosisError("Rank must be between 1 and the smaller teacher dimension")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise DiagnosisError("Seed must be a nonnegative integer below 2^31")
    if type(itq_iterations) is not int or not 0 <= itq_iterations <= 100:
        raise DiagnosisError("ITQ iterations must be 0..100")
    if type(residual) is not bool:
        raise DiagnosisError("Residual flag must be boolean")
    if type(probe_count) is not int or not 1 <= probe_count <= 512:
        raise DiagnosisError("Sample count must be 1..512")
    weight = weight.detach().to(device="cpu", dtype=torch.float32)
    out_features, in_features = weight.shape

    # Stage 1: randomized low-rank approximation before any sign conversion.
    u, v = initialize(weight, rank=rank, itq_iterations=0, seed=seed)
    lowrank = u @ v
    lowrank_relative = _relative_rmse(lowrank, weight)

    # Stage 2: the same initialization path used in the production QAT
    # trainer, including optionally another low-rank path on its residual.
    compressed = LittleBitQAT.from_teacher(
        weight, rank=rank, itq_iterations=itq_iterations, residual=residual,
        seed=seed,
    )
    compressed.eval()
    signed_weight = compressed.dense_weight()
    binary_relative = _relative_rmse(signed_weight, weight)

    # Same fixed probe seed convention as trainer audit, which enables
    # comparison to a prior result without selecting on that audit set.
    generator = torch.Generator(device="cpu").manual_seed(seed + 8892)
    probes = torch.randn(probe_count, in_features, generator=generator)
    target = probes @ weight.t()
    svd_target = probes @ lowrank.t()
    signed_target = probes @ signed_weight.t()
    lowrank_probe = _relative_rmse(svd_target, target)
    binary_probe = _relative_rmse(signed_target, target)
    if not all(math.isfinite(v) for v in (
        lowrank_relative, binary_relative, lowrank_probe, binary_probe
    )):
        raise DiagnosisError("Nonfinite approximation error; diagnostic invalid")
    return {
        "status": "PROJECTION_APPROXIMATION_DIAGNOSIS",
        "teacher_shape": [int(out_features), int(in_features)],
        "teacher_sha256": _teacher_sha(weight),
        "rank": rank,
        "seed": seed,
        "itq_iterations": itq_iterations,
        "residual_factors": residual,
        "probe_count": probe_count,
        "svd_relative_weight_rmse": lowrank_relative,
        "binary_initial_relative_weight_rmse": binary_relative,
        "svd_random_probe_nrmse": lowrank_probe,
        "binary_initial_random_probe_nrmse": binary_probe,
        "extra_binary_error_not_additive": True,
        "quantization_bpw": estimated_effective_bits(
            out_features, in_features, rank, residual),
        "estimated_effective_bpw": estimated_effective_bits(
            out_features, in_features, rank, residual),
        "lowrank_approximation_is_randomized": True,
        "initial_factors_only": True,
        "training_was_run": False,
        "full_model_trained": False,
        "measures_language_quality": False,
        "synthetic_teacher_data": None,
        "note": "Randomized rank approximation, signed initialization and random "
                "probe error are three distinct diagnostics, not guarantees "
                "of trained chatbot quality. Higher rank increases compute.",
    }


def diagnose_projection(checkpoint, *, layer=0, projection="self_attn.q_proj",
                        rank=128, seed=42, itq_iterations=20,
                        residual=True, probe_count=32, max_teacher_mib=512):
    """Load only one real-format local teacher tensor and leave all files intact."""
    weight = load_projection(checkpoint, layer=layer, projection=projection,
                             max_teacher_mib=max_teacher_mib)
    result = diagnose_matrix(
        weight, rank=rank, seed=seed, itq_iterations=itq_iterations,
        residual=residual, probe_count=probe_count)
    result.update({
        "source": str(checkpoint),
        "layer": layer,
        "projection": projection,
        "synthetic_teacher_data": False,
        "source_provenance_verified": False,
    })
    return result
