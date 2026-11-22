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
                    probe_count=32, refine_output_scales=False,
                    binary_refine_channels=0, binary_refine_passes=1,
                    svd_oversample=0, svd_power_iters=3):
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
    if type(refine_output_scales) is not bool:
        raise DiagnosisError("refine_output_scales must be boolean")
    if type(binary_refine_channels) is not int or not 0 <= binary_refine_channels <= 256:
        raise DiagnosisError("binary_refine_channels must be in 0..256")
    if type(binary_refine_passes) is not int or not 1 <= binary_refine_passes <= 8:
        raise DiagnosisError("binary_refine_passes must be in 1..8")
    if type(svd_oversample) is not int or not 0 <= svd_oversample <= 256:
        raise DiagnosisError("svd_oversample must be in 0..256")
    if type(svd_power_iters) is not int or not 0 <= svd_power_iters <= 10:
        raise DiagnosisError("svd_power_iters must be in 0..10")
    weight = weight.detach().to(device="cpu", dtype=torch.float32)
    out_features, in_features = weight.shape

    # Stage 1: randomized low-rank approximation before any sign conversion.
    u, v = initialize(weight, rank=rank, itq_iterations=0, seed=seed)
    lowrank = u @ v
    lowrank_relative = _relative_rmse(lowrank, weight)
    advanced_svd = bool(svd_oversample or svd_power_iters != 3)
    improved_lowrank = None
    if advanced_svd:
        u_advanced, v_advanced = initialize(
            weight, rank=rank, itq_iterations=0, seed=seed,
            svd_oversample=svd_oversample, svd_power_iters=svd_power_iters)
        improved_lowrank = u_advanced @ v_advanced

    # Stage 2: the same initialization path used in the production QAT
    # trainer, including optionally another low-rank path on its residual.
    compressed = LittleBitQAT.from_teacher(
        weight, rank=rank, itq_iterations=itq_iterations, residual=residual,
        seed=seed,
    )
    compressed.eval()
    signed_weight = compressed.dense_weight()
    improved_binary_weight = None
    if advanced_svd:
        compressed = LittleBitQAT.from_teacher(
            weight, rank=rank, itq_iterations=itq_iterations, residual=residual,
            seed=seed, svd_oversample=svd_oversample,
            svd_power_iters=svd_power_iters)
        compressed.eval()
        improved_binary_weight = compressed.dense_weight()
    refined_weight = None
    refinement_details = None
    if refine_output_scales:
        from .scale_refinement import refine_output_row_scales
        refinement_details = refine_output_row_scales(compressed, weight)
        refined_weight = compressed.dense_weight()
    binary_sign_weight = None
    sign_details = None
    if binary_refine_channels:
        from .binary_refinement import refine_binary_signs
        sign_details = refine_binary_signs(
            compressed, weight, channels=binary_refine_channels,
            passes=binary_refine_passes)
        binary_sign_weight = compressed.dense_weight()
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
    improved_lowrank_relative = (_relative_rmse(improved_lowrank, weight)
                                 if improved_lowrank is not None else None)
    improved_lowrank_probe = (_relative_rmse(probes @ improved_lowrank.t(), target)
                              if improved_lowrank is not None else None)
    improved_binary_relative = (_relative_rmse(improved_binary_weight, weight)
                                if improved_binary_weight is not None else None)
    improved_binary_probe = (_relative_rmse(probes @ improved_binary_weight.t(), target)
                             if improved_binary_weight is not None else None)
    refined_relative = (_relative_rmse(refined_weight, weight)
                        if refined_weight is not None else None)
    refined_probe = (_relative_rmse(probes @ refined_weight.t(), target)
                     if refined_weight is not None else None)
    signed_refined_relative = (_relative_rmse(binary_sign_weight, weight)
                               if binary_sign_weight is not None else None)
    signed_refined_probe = (_relative_rmse(probes @ binary_sign_weight.t(), target)
                            if binary_sign_weight is not None else None)
    values = [lowrank_relative, binary_relative, lowrank_probe, binary_probe]
    values += [x for x in (refined_relative, refined_probe,
                          signed_refined_relative, signed_refined_probe,
                          improved_lowrank_relative, improved_lowrank_probe,
                          improved_binary_relative, improved_binary_probe) if x is not None]
    if not all(math.isfinite(v) for v in values):
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
        "svd_oversample": svd_oversample,
        "svd_power_iters": svd_power_iters,
        "svd_enhanced_relative_weight_rmse": improved_lowrank_relative,
        "svd_enhanced_random_probe_nrmse": improved_lowrank_probe,
        "binary_svd_enhanced_relative_weight_rmse": improved_binary_relative,
        "binary_svd_enhanced_random_probe_nrmse": improved_binary_probe,
        "refine_output_scales": refine_output_scales,
        "binary_refined_relative_weight_rmse": refined_relative,
        "binary_refined_random_probe_nrmse": refined_probe,
        "binary_refine_channels": binary_refine_channels,
        "binary_refine_passes": binary_refine_passes,
        "binary_sign_refined_relative_weight_rmse": signed_refined_relative,
        "binary_sign_refined_random_probe_nrmse": signed_refined_probe,
        "binary_refine_details": sign_details,
        "refinement_weight_sse_before": (
            refinement_details["weight_sse_before"] if refinement_details else None),
        "refinement_weight_sse_after": (
            refinement_details["weight_sse_after"] if refinement_details else None),
        "refinement_changed": (
            refinement_details["changed"] if refinement_details else None),
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
                        residual=True, probe_count=32, max_teacher_mib=512,
                        refine_output_scales=False, binary_refine_channels=0,
                        binary_refine_passes=1, svd_oversample=0,
                        svd_power_iters=3):
    """Load only one real-format local teacher tensor and leave all files intact."""
    weight = load_projection(checkpoint, layer=layer, projection=projection,
                             max_teacher_mib=max_teacher_mib)
    result = diagnose_matrix(
        weight, rank=rank, seed=seed, itq_iterations=itq_iterations,
        residual=residual, probe_count=probe_count,
        refine_output_scales=refine_output_scales,
        binary_refine_channels=binary_refine_channels,
        binary_refine_passes=binary_refine_passes,
        svd_oversample=svd_oversample,
        svd_power_iters=svd_power_iters)
    result.update({
        "source": str(checkpoint),
        "layer": layer,
        "projection": projection,
        "synthetic_teacher_data": False,
        "source_provenance_verified": False,
    })
    return result
