"""Offline, sequential LittleBit-2 compression of local dense Gemma projections.

Never downloads a checkpoint; never certifies conversational quality.
Scans Safetensors HEADER metadata first, trains exactly one projection at a
time, writes verified packed weights and an atomic, checksummed progress
manifest. Resume refuses changed settings/teacher or modified artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from safetensors import safe_open

from .layer_train import PROJECTIONS, PREFIX, TrainingError, _teacher_sha, load_projection

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4}


class BatchTrainingError(TrainingError):
    """Invalid checkpoint, stale batch manifest or exceeded safety budget."""


def _validate_selection(layer_start, layer_end, projections):
    if (type(layer_start) is not int or type(layer_end) is not int
            or not 0 <= layer_start < layer_end <= 62):
        raise BatchTrainingError("Layer selection must be a nonempty range within 0..62")
    if (not isinstance(projections, (tuple, list)) or not projections
            or len(set(projections)) != len(projections)
            or any(name not in PROJECTIONS for name in projections)):
        raise BatchTrainingError("Select distinct Gemma projection names")
    return tuple(projections)


def _safe_json(path):
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise BatchTrainingError("Missing or oversized local Safetensors index")
    try:
        result = json.loads(path.read_text(encoding="utf8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchTrainingError("Malformed JSON checkpoint index") from exc
    if not isinstance(result, dict):
        raise BatchTrainingError("Safetensors index must be a JSON object")
    return result


def _read_inventory(source, layer_start, layer_end, projections, max_teacher_mib,
                    allow_test_shapes):
    from engine.gemma3_bridge import read_gemma_config, test_fixture_path, projection_spec
    source = Path(source)
    if not source.exists():
        raise BatchTrainingError("Missing local dense teacher checkpoint: " + str(source))
    if type(max_teacher_mib) is not int or not 1 <= max_teacher_mib <= 4096:
        raise BatchTrainingError("Selected teacher tensor budget must be 1..4096 MiB")
    shapes = projection_spec(read_gemma_config(test_fixture_path()))
    if source.is_file():
        if source.suffix != ".safetensors":
            raise BatchTrainingError("Checkpoint file must be a .safetensors file")
        weight_map = None
    elif source.is_dir():
        index = _safe_json(source / "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if (not isinstance(weight_map, dict) or
                any(not isinstance(k, str) or not isinstance(v, str)
                    for k, v in weight_map.items())):
            raise BatchTrainingError("Invalid sharded checkpoint weight_map")
    else:
        raise BatchTrainingError("Checkpoint is not a file or directory")

    # Open shards only as needed; safetensors.get_slice reads metadata/shape,
    # not the actual dense matrix buffers.
    checked_shards = {}
    result = []
    for index in range(layer_start, layer_end):
        for projection in projections:
            key = f"{PREFIX}.{index}.{projection}.weight"
            if weight_map is not None:
                if key not in weight_map:
                    raise BatchTrainingError("Missing dense teacher projection: " + key)
                filename = weight_map[key]
                if (not filename.endswith(".safetensors")
                        or filename != Path(filename).name
                        or "/" in filename or "\\" in filename
                        or filename in (".", "..")):
                    raise BatchTrainingError("Unsafe teacher shard filename: " + filename)
                target = source / filename
                if not target.resolve().is_relative_to(source.resolve()):
                    raise BatchTrainingError("Unsafe teacher shard path: " + filename)
            else:
                target = source
            if not target.is_file():
                raise BatchTrainingError("Missing dense teacher shard: " + str(target))
            # Reopening the header is inexpensive, and avoids keeping file
            # handles open across large training jobs.
            try:
                with safe_open(str(target), framework="np", device="cpu") as opened:
                    if key not in opened.keys():
                        raise BatchTrainingError("Missing dense teacher tensor in shard: " + key)
                    tensor_slice = opened.get_slice(key)
                    actual_shape = list(tensor_slice.get_shape())
                    dtype = tensor_slice.get_dtype()
            except (OSError, ValueError) as exc:
                if isinstance(exc, BatchTrainingError):
                    raise
                raise BatchTrainingError("Cannot read teacher Safetensors header: " + str(target)) from exc
            if dtype not in DTYPE_BYTES or len(actual_shape) != 2:
                raise BatchTrainingError("Unsupported teacher matrix dtype/shape: " + key)
            if any(type(dim) is not int or dim < 1 for dim in actual_shape):
                raise BatchTrainingError("Invalid teacher matrix dimensions: " + key)
            original_bytes = actual_shape[0] * actual_shape[1] * DTYPE_BYTES[dtype]
            if original_bytes > max_teacher_mib * 1024 * 1024:
                raise BatchTrainingError("Teacher matrix exceeds per-projection RAM budget: " + key)
            true_shape = list(shapes[projection])
            matches = actual_shape == true_shape
            if not matches and not allow_test_shapes:
                raise BatchTrainingError(
                    "Teacher shape does not match Gemma 3 27B: " + key
                    + f" expected {true_shape}, got {actual_shape}")
            result.append({
                "layer": index, "projection": projection, "key": key,
                "shape": actual_shape, "dtype": dtype,
                "teacher_tensor_bytes": original_bytes,
                "source_shard": str(target),
                "matches_gemma3_27b_shape": matches,
            })
    return result


def scan_source(source, *, layer_start=0, layer_end=62, projections=PROJECTIONS,
                max_teacher_mib=512, allow_test_shapes=False):
    """Local-only inventory of selected original dense Gemma checkpoint keys."""
    projections = _validate_selection(layer_start, layer_end, projections)
    found = _read_inventory(source, layer_start, layer_end, projections,
                            max_teacher_mib, allow_test_shapes)
    full_set = (layer_start == 0 and layer_end == 62
                and set(projections) == set(PROJECTIONS))
    return {
        "status": "DENSE_TEACHER_ALL_434_MATRICES" if full_set else "DENSE_TEACHER_PARTIAL",
        "source": str(Path(source).resolve()),
        "requested_projections": (layer_end - layer_start) * len(projections),
        "available_projections": len(found),
        "full_size_shapes": all(x["matches_gemma3_27b_shape"] for x in found),
        "source_is_pretrained_verified": False,
        "trained_littlebit_weights_present": False,
        "full_model_trained": False,
        "no_model_weights_materialized": True,
        "projections": found,
    }


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if path.is_symlink() or tmp.is_symlink():
        raise BatchTrainingError("Refusing symlink batch manifest")
    with tmp.open("w", encoding="utf8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _settings(source, layer_start, layer_end, projections, *, rank, steps,
              batch_size, lr, itq_iterations, seed, device, residual,
              max_teacher_mib, checkpoint_every, max_holdout_nrmse,
              allow_test_shapes, refine_output_scales=False,
              binary_refine_channels=0, binary_refine_passes=1,
              svd_oversample=0, svd_power_iters=3):
    result = {
        "source": str(Path(source).resolve()), "layer_start": layer_start,
        "layer_end": layer_end, "projections": list(projections),
        "rank": rank, "steps": steps, "batch_size": batch_size, "lr": lr,
        "itq_iterations": itq_iterations, "seed": seed, "device": device,
        "residual": residual, "max_teacher_mib": max_teacher_mib,
        "checkpoint_every": checkpoint_every,
        "max_holdout_nrmse": max_holdout_nrmse,
        "allow_test_shapes": allow_test_shapes,
    }
    # Keep old default-mode manifests resumable, including pre-feature jobs.
    if refine_output_scales:
        result["refine_output_scales"] = True
    if binary_refine_channels:
        result["binary_refine_channels"] = binary_refine_channels
    # Omit legacy defaults so pre-upgrade batch manifests remain resumable.
    if binary_refine_passes != 1:
        result["binary_refine_passes"] = binary_refine_passes
    if svd_oversample:
        result["svd_oversample"] = svd_oversample
    if svd_power_iters != 3:
        result["svd_power_iters"] = svd_power_iters
    return result


def _resume_audit(record, entry, out, source, settings):
    # Manifests are untrusted data; derive names from the requested selection,
    # not from a previously saved arbitrary filesystem path.
    stem = f"layer_{entry['layer']:02d}_{entry['projection'].replace('.', '_')}"
    expected = out / (stem + f"_step_{settings['steps']:06d}.safetensors")
    audit_file = out / (stem + f"_step_{settings['steps']:06d}.json")
    if (record.get("packed_file") != str(expected)
            or record.get("audit_json") != str(audit_file)
            or not record.get("packed_inference_verified")):
        raise BatchTrainingError("Resume record configuration mismatch: " + stem)
    if (not expected.is_file() or not audit_file.is_file()
            or expected.is_symlink() or audit_file.is_symlink()):
        raise BatchTrainingError("Resume output is missing or unsafe: " + stem)
    if _sha_file(expected) != record.get("packed_sha256"):
        raise BatchTrainingError("Resume checksum indicates modified/corrupt packed output: " + stem)
    audit = _safe_json(audit_file)
    if (audit.get("teacher_sha256") != record.get("teacher_sha256")
            or audit.get("rank") != record.get("rank")
            or audit.get("packed_file") != str(expected)
            or audit.get("refine_output_scales", False) != settings.get("refine_output_scales", False)
            or record.get("refine_output_scales", False) != settings.get("refine_output_scales", False)
            or audit.get("binary_refine_channels", 0) != settings.get("binary_refine_channels", 0)
            or record.get("binary_refine_channels", 0) != settings.get("binary_refine_channels", 0)
            or audit.get("binary_refine_passes", 1) != settings.get("binary_refine_passes", 1)
            or record.get("binary_refine_passes", 1) != settings.get("binary_refine_passes", 1)
            or audit.get("svd_oversample", 0) != settings.get("svd_oversample", 0)
            or record.get("svd_oversample", 0) != settings.get("svd_oversample", 0)
            or audit.get("svd_power_iters", 3) != settings.get("svd_power_iters", 3)
            or record.get("svd_power_iters", 3) != settings.get("svd_power_iters", 3)
            or not audit.get("packed_inference_verified")):
        raise BatchTrainingError("Resume audit does not match checkpoint manifest: " + stem)
    # Verify original dense teacher has not changed. This is intentionally
    # one projection's CPU tensor at a time, not the whole source model.
    teacher = load_projection(source, layer=entry["layer"],
                              projection=entry["projection"],
                              max_teacher_mib=settings["max_teacher_mib"])
    if _teacher_sha(teacher) != record["teacher_sha256"]:
        raise BatchTrainingError("Resume teacher tensor has changed: " + stem)


def random_probe_gate(record, threshold):
    """Pass only a finite, nonregressing random-probe audit and packed parity.

    This is NOT a language quality metric. The held-out audit probes are
    independent of the validation probes used to select the QAT checkpoint.
    """
    if (type(threshold) not in (float, int) or not math.isfinite(threshold)
            or not 0 < threshold <= 10):
        raise BatchTrainingError("Invalid quality gate threshold")
    try:
        normalized = float(record["holdout_nrmse"])
        current = float(record["holdout_rmse"])
        initial = float(record["initial_holdout_rmse"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (record.get("packed_inference_verified") is True
            and all(math.isfinite(x) for x in (normalized, current, initial))
            and normalized <= threshold and current <= initial + 1e-8)


def run_batch(*, source, out, layer_start=0, layer_end=62,
              projections=PROJECTIONS, rank=None, steps=50, batch_size=16,
              lr=0.002, itq_iterations=20, seed=42, device="cpu", residual=True,
              max_teacher_mib=512, checkpoint_every=10, max_jobs=1,
              max_holdout_nrmse=0.25, allow_test_shapes=False, resume=False,
              refine_output_scales=False, binary_refine_channels=0,
              binary_refine_passes=1, svd_oversample=0, svd_power_iters=3):
    """Sequential train/verify/export with a durable checksummed progress file.

    Safe by default: at most one job per invocation, one teacher projection
    resident at a time, and NO full-model-ready claim. Increase max_jobs
    deliberately for lengthy sweeps. Never overwrites a completed export.
    """
    if type(max_jobs) is not int or not 1 <= max_jobs <= 434:
        raise BatchTrainingError("max_jobs must be in 1..434")
    if (type(max_holdout_nrmse) not in (float, int)
            or not math.isfinite(max_holdout_nrmse)
            or not 0 < max_holdout_nrmse <= 10):
        raise BatchTrainingError("Invalid quality gate max_holdout_nrmse")
    if (type(resume) is not bool or type(allow_test_shapes) is not bool
            or type(refine_output_scales) is not bool):
        raise BatchTrainingError("Invalid boolean batch setting")
    if type(binary_refine_channels) is not int or not 0 <= binary_refine_channels <= 256:
        raise BatchTrainingError("binary_refine_channels must be in 0..256")
    if type(binary_refine_passes) is not int or not 1 <= binary_refine_passes <= 8:
        raise BatchTrainingError("binary_refine_passes must be in 1..8")
    if type(svd_oversample) is not int or not 0 <= svd_oversample <= 256:
        raise BatchTrainingError("svd_oversample must be in 0..256")
    if type(svd_power_iters) is not int or not 0 <= svd_power_iters <= 10:
        raise BatchTrainingError("svd_power_iters must be in 0..10")

    projections = _validate_selection(layer_start, layer_end, projections)
    selected = _read_inventory(source, layer_start, layer_end, projections,
                                max_teacher_mib, allow_test_shapes)
    params = _settings(source, layer_start, layer_end, projections, rank=rank,
                       steps=steps, batch_size=batch_size, lr=lr,
                       itq_iterations=itq_iterations, seed=seed, device=device,
                       residual=residual, max_teacher_mib=max_teacher_mib,
                       checkpoint_every=checkpoint_every,
                       max_holdout_nrmse=max_holdout_nrmse,
                       allow_test_shapes=allow_test_shapes,
                       refine_output_scales=refine_output_scales,
                       binary_refine_channels=binary_refine_channels,
            binary_refine_passes=binary_refine_passes,
            svd_oversample=svd_oversample,
            svd_power_iters=svd_power_iters,
                       binary_refine_passes=binary_refine_passes,
                       svd_oversample=svd_oversample,
                       svd_power_iters=svd_power_iters)
    raw_out = Path(out)
    if raw_out.is_symlink():
        raise BatchTrainingError("Refusing output directory symlink")
    out = raw_out.resolve()
    manifest_path = out / "batch_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise BatchTrainingError("Existing batch output requires --resume; no overwrite")
        manifest = _safe_json(manifest_path)
        if manifest.get("schema_version") != 1 or manifest.get("settings") != params:
            raise BatchTrainingError("Batch configuration mismatch on resume")
        jobs = manifest.get("jobs")
        if not isinstance(jobs, dict):
            raise BatchTrainingError("Invalid batch progress record")
    else:
        if resume:
            raise BatchTrainingError("Cannot resume: batch_manifest.json is missing")
        out.mkdir(parents=True, exist_ok=True)
        jobs = {}
        manifest = {"schema_version": 1, "settings": params, "jobs": jobs,
                    "full_model_trained": False, "validated_language_quality": False,
                    "ready_for_chat": False}
        _write_atomic(manifest_path, manifest)

    # Import after up-front checkpoint/settings validation to avoid accidental
    # training or side effects when source shape, rank or manifest is invalid.
    from .cli import run as run_single
    skipped = 0
    new = 0
    for entry in selected:
        key = f"{entry['layer']:02d}:{entry['projection']}"
        if key in jobs:
            _resume_audit(jobs[key], entry, out, source, params)
            skipped += 1
            continue
        if new >= max_jobs:
            break
        stem = f"layer_{entry['layer']:02d}_{entry['projection'].replace('.', '_')}"
        state = out / (stem + ".state.pt")
        single_args = SimpleNamespace(
            mode="layer", checkpoint=Path(source), out=out,
            layer=entry["layer"], projection=entry["projection"],
            rank=rank, eff_bit=0.55, steps=steps,
            batch_size=batch_size, lr=lr, itq_iterations=itq_iterations,
            seed=seed, device=device, no_residual=not residual,
            max_teacher_mib=max_teacher_mib, checkpoint_every=checkpoint_every,
            resume=(resume and state.is_file()),
            refine_output_scales=refine_output_scales,
            binary_refine_channels=binary_refine_channels,
        )
        print("Training " + key + " (new " + str(new + 1) + "/" + str(min(max_jobs, len(selected))) + ")",
              file=sys.stderr, flush=True)
        record = run_single(single_args)
        packed = Path(record["packed_file"])
        record["packed_sha256"] = _sha_file(packed)
        record["passes_random_probe_gate"] = random_probe_gate(
            record, max_holdout_nrmse)
        record["holdout_error_regressed"] = (
            float(record["holdout_rmse"]) > float(record["initial_holdout_rmse"]) + 1e-8)
        record["random_probe_gate_threshold"] = float(max_holdout_nrmse)
        record["language_quality_verified"] = False
        record["full_model_trained"] = False
        jobs[key] = record
        _write_atomic(manifest_path, manifest)
        new += 1
    ordered = [jobs[f"{e['layer']:02d}:{e['projection']}"]
               for e in selected if f"{e['layer']:02d}:{e['projection']}" in jobs]
    return {
        "status": "BATCH_LAYER_QAT_PARTIAL",
        "source": str(Path(source).resolve()), "out": str(out),
        "manifest": str(manifest_path),
        "total_jobs": len(selected), "completed_jobs": len(ordered),
        "new_jobs": new, "skipped_jobs": skipped,
        "random_probe_gate_passed": sum(bool(j["passes_random_probe_gate"]) for j in ordered),
        "random_probe_gate_failed": sum(not bool(j["passes_random_probe_gate"]) for j in ordered),
        "full_model_trained": False, "ready_for_chat": False,
        "validated_language_quality": False,
        "teacher_pretraining_verified": False,
        "note": "One dense projection at a time; random-probe fit is NOT language quality.",
        "jobs": ordered,
    }
