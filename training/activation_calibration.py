"""Activation-aware, same-BPW calibration of existing LittleBit signed factors.

Requires an externally captured, metadata-bound three-way activation bundle:
calibration fits fixed output-row scales, selection decides deployment/fallback,
and an untouched audit reports performance without selecting candidates.

This module never invents activations, downloads model weights, or certifies
whether a self-declared capture was genuinely produced by the original model.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .layer_train import _teacher_sha, estimated_effective_bits


class ActivationDataError(ValueError):
    """Missing, unsafe, or incompatible external activation capture."""


@dataclass(frozen=True)
class ActivationBundle:
    calibration: torch.Tensor
    selection: torch.Tensor
    audit: torch.Tensor
    metadata: dict
    activation_sha256: str
    provenance_independently_verified: bool = False


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def _sample_hashes(tensor):
    """Detect exact cross-split row reuse, not just identical whole files."""
    return {
        hashlib.sha256(row.contiguous().numpy().tobytes()).digest()
        for row in tensor
    }


def load_activation_bundle(source, teacher, *, layer, projection, max_mib=256):
    """Load an explicit user-provided Safetensors forward-pre-hook capture.

    Binding checks prevent accidentally mixing different layer projections
    or different local teacher tensors. Declared provenance is NOT proof
    that the original language model was actually executed.
    """
    if type(max_mib) is not int or not 1 <= max_mib <= 4096:
        raise ActivationDataError("max_mib must be in 1..4096")
    if type(layer) is not int or not 0 <= layer < 62:
        raise ActivationDataError("Invalid layer index")
    if not isinstance(projection, str) or not projection:
        raise ActivationDataError("A target projection is required")
    if not isinstance(teacher, torch.Tensor) or teacher.ndim != 2:
        raise ActivationDataError("Expected one dense teacher matrix")
    if not bool(torch.isfinite(teacher).all()):
        raise ActivationDataError("Teacher must be finite")

    path = Path(source)
    if (not path.is_file() or path.is_symlink()
            or path.suffix != ".safetensors"):
        raise ActivationDataError("Activation capture file missing or unsafe: "+str(path))
    if path.stat().st_size > max_mib * 1024 * 1024:
        raise ActivationDataError("Activation capture exceeds configured MiB limit")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            keys = set(opened.keys())
            if keys != {"calibration", "selection", "audit"}:
                raise ActivationDataError(
                    "Activation file must contain calibration, selection and audit only")
            metadata = dict(opened.metadata() or {})
            expected = {
                "teacher_sha256": _teacher_sha(teacher),
                "layer": str(layer),
                "projection": projection,
                "input_module": f"language_model.model.layers.{layer}.{projection}",
                "split_method": "disjoint_prompts",
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ActivationDataError("Activation provenance/binding mismatch: "+key)
            method = metadata.get("capture_method")
            if method == "forward_pre_hook":
                pass
            elif method == "gemma3_layer0_embedding_rmsnorm":
                if (layer != 0 or projection not in (
                        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
                        or metadata.get("embedding_tensor") != (
                            "language_model.model.embed_tokens.weight")
                        or metadata.get("normalization_tensor") != (
                            "language_model.model.layers.0.input_layernorm.weight")):
                    raise ActivationDataError(
                        "Invalid lightweight layer-0 capture provenance")
            else:
                raise ActivationDataError("Unsupported activation capture_method/provenance")
            if not metadata.get("source_model", "").strip():
                raise ActivationDataError("Activation provenance missing source_model")
            split_hashes = []
            for name in ("calibration", "selection", "audit"):
                sha = metadata.get(name+"_text_sha256")
                if not _valid_sha(sha):
                    raise ActivationDataError("Missing valid source prompt SHA: "+name)
                split_hashes.append(sha)
            if len(set(split_hashes)) != 3:
                raise ActivationDataError("Calibration/selection/audit prompt split overlap")
            arrays = {name: opened.get_tensor(name).float().contiguous()
                      for name in ("calibration", "selection", "audit")}
    except (OSError, RuntimeError) as exc:
        raise ActivationDataError("Unable to read activation capture: "+str(exc)) from exc

    width = teacher.shape[1]
    for name, samples in arrays.items():
        if (samples.ndim != 2 or samples.shape[1] != width
                or not 16 <= samples.shape[0] <= 8192):
            raise ActivationDataError(name+" must contain 16..8192 rows of teacher input width")
        if not bool(torch.isfinite(samples).all()):
            raise ActivationDataError(name+" must contain only finite activations")

    hashes = {key: _sample_hashes(samples) for key, samples in arrays.items()}
    for left, right in (("calibration", "selection"),
                        ("calibration", "audit"), ("selection", "audit")):
        # Shared BOS/prefix token inputs can legitimately recur across
        # disjoint prompts. Reject fully reused splits, not shared prefixes.
        if hashes[left].issubset(hashes[right]) or hashes[right].issubset(hashes[left]):
            raise ActivationDataError("Whole activation split overlap: "+left+"/"+right)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return ActivationBundle(arrays["calibration"], arrays["selection"], arrays["audit"],
                            metadata, digest)


@torch.no_grad()
def calibrate_activation_scales(model, teacher, calibration, selection, audit,
                                *, min_relative_improvement=0.0):
    """Refit only output-row scales and gate on separate selection activations.

    The input activations are supplied by the caller, never Gaussian-generated.
    For each output row, solve nonnegative two-path least squares for
    || X W^T - g0 X W0^T - g1 X W1^T ||^2.
    The audit split is evaluated AFTER the selection decision; it cannot
    influence acceptance. A rejected candidate restores every original scale.
    """
    if (type(min_relative_improvement) not in (int, float)
            or not math.isfinite(min_relative_improvement)
            or not 0 <= min_relative_improvement < 1):
        raise ValueError("min_relative_improvement must be a number in [0,1)")
    if not isinstance(teacher, torch.Tensor) or teacher.ndim != 2:
        raise ValueError("Expected 2D dense teacher")
    if not bool(torch.isfinite(teacher).all()):
        raise ValueError("Teacher must be finite")
    device=model.u1.device
    target_weight = teacher.detach().to(dtype=torch.float32, device=device)
    if tuple(target_weight.shape) != tuple(model.dense_weight().shape):
        raise ValueError("Teacher shape differs from compressed projection")
    widths = target_weight.shape[1]
    for name, xs in (("calibration", calibration), ("selection", selection), ("audit", audit)):
        if (not isinstance(xs, torch.Tensor) or xs.ndim != 2
                or xs.shape[1] != widths or xs.shape[0] < 1
                or not bool(torch.isfinite(xs).all())):
            raise ValueError(name+" activations have invalid shape or values")
    xcal = calibration.detach().to(device=device, dtype=torch.float32)
    xsel = selection.detach().to(device=device, dtype=torch.float32)
    xaudit = audit.detach().to(device=device, dtype=torch.float32)

    before_bits = estimated_effective_bits(
        target_weight.shape[0], widths, model.U.shape[1], model.residual)
    saved_primary = model.u1.detach().clone()
    saved_residual = model.u1_R.detach().clone() if model.residual else None

    def restore():
        model.u1.copy_(saved_primary)
        if saved_residual is not None:
            model.u1_R.copy_(saved_residual)

    def score(xs):
        truth = xs @ target_weight.t()
        estimate = model(xs)
        mse = float((estimate - truth).square().mean())
        reference = float(truth.square().mean())
        if not math.isfinite(mse) or not math.isfinite(reference):
            raise ValueError("Nonfinite activation reconstruction error")
        return float(math.sqrt(mse) / max(math.sqrt(reference), 1e-12))

    cal_before = score(xcal)
    sel_before = score(xsel)
    audit_before = score(xaudit)
    y = (xcal @ target_weight.t()).double()
    a = model._forward_path(xcal, "").double()
    b = model._forward_path(xcal, "_R").double() if model.residual else None
    aa, ay = (a*a).sum(dim=0), (a*y).sum(dim=0)
    zero = torch.zeros_like(aa)
    one = torch.ones_like(aa)
    safe_aa = torch.where(aa > 1e-30, aa, one)
    best_a = one.clone()
    best_b = one.clone() if model.residual else zero.clone()
    bb = (b*b).sum(dim=0) if model.residual else zero
    ab = (a*b).sum(dim=0) if model.residual else zero
    by = (b*y).sum(dim=0) if model.residual else zero

    # Compare SSE minus constant ||y||^2, per output row.
    def cost(p, q):
        return aa*p*p + bb*q*q + 2*ab*p*q - 2*ay*p - 2*by*q

    best_cost = cost(best_a, best_b)

    def offer(p, q, allowed):
        nonlocal best_a, best_b, best_cost
        candidate_cost = cost(p, q)
        take = (allowed & (p >= 0) & (q >= 0)
                & torch.isfinite(candidate_cost) & torch.isfinite(p)
                & torch.isfinite(q) & (candidate_cost < best_cost))
        best_a = torch.where(take, p, best_a)
        best_b = torch.where(take, q, best_b)
        best_cost = torch.where(take, candidate_cost, best_cost)

    offer(zero, zero, torch.ones_like(aa, dtype=torch.bool))
    offer((ay/safe_aa).clamp_min(0), zero, aa > 1e-30)
    if model.residual:
        safe_bb = torch.where(bb > 1e-30, bb, one)
        offer(zero, (by/safe_bb).clamp_min(0), bb > 1e-30)
        det = aa*bb - ab.square()
        valid = det > (aa*bb).clamp_min(1e-30) * 1e-10
        safe_det = torch.where(valid, det, one)
        p = (ay*bb - by*ab)/safe_det
        q = (by*aa - ay*ab)/safe_det
        offer(p, q, valid)

    next_a = saved_primary * best_a.to(saved_primary.dtype)
    next_b = saved_residual * best_b.to(saved_residual.dtype) if model.residual else None
    if not bool(torch.isfinite(next_a).all()) or (
            next_b is not None and not bool(torch.isfinite(next_b).all())):
        raise ValueError("Nonfinite activation-fitted output-row scales")
    model.u1.copy_(next_a)
    if next_b is not None:
        model.u1_R.copy_(next_b)

    cal_candidate = score(xcal)
    sel_candidate = score(xsel)
    accepted = (cal_candidate <= cal_before + 1e-6
                and sel_candidate < sel_before * (1-min_relative_improvement) - 1e-9)
    after_bits = estimated_effective_bits(
        target_weight.shape[0], widths, model.U.shape[1], model.residual)
    if after_bits != before_bits:
        restore()
        raise ValueError("Physical BPW changed: candidate rejected")
    # The untouched audit is NEVER used for the selection decision.
    if not accepted:
        restore()
    cal_after = score(xcal)
    sel_after = score(xsel)
    audit_after = score(xaudit)
    if not accepted and (abs(sel_after-sel_before) > 1e-8
                         or abs(audit_after-audit_before) > 1e-8):
        raise ValueError("Failed to restore rejected candidate exactly")
    return {
        "decision": "ACTIVATION_CANDIDATE_ACCEPTED" if accepted else "FALLBACK_LEGACY",
        "accepted": bool(accepted),
        "calibration_nrmse_before": cal_before,
        "calibration_nrmse_candidate": cal_candidate,
        "calibration_nrmse_after": cal_after,
        "selection_nrmse_before": sel_before,
        "selection_nrmse_candidate": sel_candidate,
        "selection_nrmse_after": sel_after,
        "audit_nrmse_before": audit_before,
        "audit_nrmse_after": audit_after,
        "audit_independent_of_selection": True,
        "estimated_effective_bpw": before_bits,
        "same_bpw_verified": bool(before_bits == after_bits),
        "min_relative_improvement": float(min_relative_improvement),
        "optimizes_only_existing_output_row_scales": True,
        "validated_language_quality": False,
    }
