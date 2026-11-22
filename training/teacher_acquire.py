"""Acquire only the *original pretrained* Gemma 3 shard needed for one projection.

This code never bypasses the official Google's gated-model license. It uses
Hugging Face's already authenticated local token, pins a commit revision,
downloads just the index for planning, and requires --download for weights.
The single weight shard is not a complete pretrained 27B teacher model.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

from .layer_train import PREFIX, PROJECTIONS

OFFICIAL_GEMMA_REPO = "google/gemma-3-27b-it"
OFFICIAL_GEMMA_PAGE = "https://huggingface.co/google/gemma-3-27b-it"
INDEX_NAME = "model.safetensors.index.json"
MAX_INDEX_BYTES = 16 * 1024 * 1024


class TeacherAcquireError(ValueError):
    """License, index, provenance, disk budget or download failure."""


def _valid_shard_filename(filename):
    return (isinstance(filename, str) and filename == Path(filename).name
            and "/" not in filename and "\\" not in filename
            and filename.startswith("model-") and filename.endswith(".safetensors")
            and not filename.startswith(".") and len(filename) <= 255)


def plan_from_index(index_data, layer, projection):
    """Resolve exact original Gemma dense projection to one shard by index."""
    if type(layer) is not int or not 0 <= layer < 62:
        raise ValueError("Gemma 3 27B decoder layer must be 0..61")
    if projection not in PROJECTIONS:
        raise ValueError("Unsupported Gemma 3 projection name")
    key = f"{PREFIX}.{layer}.{projection}.weight"
    if not isinstance(index_data, dict) or not isinstance(index_data.get("weight_map"), dict):
        raise TeacherAcquireError("Missing Safetensors weight_map in original checkpoint index")
    filename = index_data["weight_map"].get(key)
    if filename is None:
        raise TeacherAcquireError("Missing original dense Gemma projection in index: " + key)
    if not _valid_shard_filename(filename):
        raise TeacherAcquireError("Unsafe or unsupported original Gemma shard filename")
    return {
        "tensor_key": key,
        "shard_filename": filename,
        "complete_model_download": False,
    }


def _require_safe_root(path):
    root = Path(path).expanduser()
    if root.is_symlink() or any(p.is_symlink() for p in list(root.parents)[:4]):
        raise TeacherAcquireError("Refusing a symlink as teacher output directory")
    root = root.resolve()
    if root.exists() and not root.is_dir():
        raise TeacherAcquireError("Teacher destination exists and is not a directory")
    return root


def _download_within_root(downloader, *, root, filename, revision):
    try:
        path = Path(downloader(
            repo_id=OFFICIAL_GEMMA_REPO, filename=filename,
            revision=revision, local_dir=str(root)
        ))
    except Exception as exc:
        raise TeacherAcquireError(
            "Official Gemma download was denied or failed. Open "
            + OFFICIAL_GEMMA_PAGE
            + " while logged in, accept Google's Gemma license, and run "
              "'hf auth login' locally. Check your network connection."
        ) from exc
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise TeacherAcquireError("Downloaded artifact path is missing or outside selected model directory")
    return path


def acquire_teacher_shard(*, out, layer=0, projection="self_attn.q_proj",
                          download=False, max_shard_gib=8, revision="main",
                          model_info=None, hf_download=None):
    """Plan first, optionally fetch one original BF16 shard from Google.

    Tests supply model_info/hf_download mocks and do not access the internet.
    Under normal usage huggingface_hub reads the user-entered HF credentials
    from its local CLI login cache; this program never handles raw tokens.
    """
    if type(download) is not bool:
        raise ValueError("download flag must be boolean")
    if (type(max_shard_gib) not in (float, int) or not math.isfinite(max_shard_gib)
            or not 0 < max_shard_gib <= 64):
        raise ValueError("max-shard-gib must be a finite amount in (0,64]")
    if revision != "main":
        raise ValueError("Only the official default Gemma main revision is supported")
    plan_from_index({"weight_map": {f"{PREFIX}.{layer}.{projection}.weight": "model-00001.safetensors"}},
                    layer, projection)
    root = _require_safe_root(out)
    # Don't import HF auth/network code for the existing offline unit suite.
    if model_info is None or hf_download is None:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise TeacherAcquireError(
                "Install the optional official Hugging Face downloader: "
                "python -m pip install -r requirements-teacher.txt"
            ) from exc
        if model_info is None:
            model_info = HfApi().model_info
        if hf_download is None:
            hf_download = hf_hub_download
    try:
        info = model_info(repo_id=OFFICIAL_GEMMA_REPO, revision="main",
                          files_metadata=True)
    except Exception as exc:
        raise TeacherAcquireError(
            "Cannot access official Gemma 3 27B metadata. Visit "
            + OFFICIAL_GEMMA_PAGE
            + " and accept Google's model license using your Hugging Face "
              "account, then run 'hf auth login'. No model weights were downloaded."
        ) from exc
    sha = getattr(info, "sha", "")
    if not isinstance(sha, str) or not re.fullmatch("[a-fA-F0-9]{40}", sha):
        raise TeacherAcquireError("Official repository metadata did not provide a valid pinned revision")
    siblings = getattr(info, "siblings", None)
    if not isinstance(siblings, (tuple, list)):
        raise TeacherAcquireError("Official repository file inventory missing")
    root.mkdir(parents=True, exist_ok=True)
    index_path = _download_within_root(hf_download, root=root, filename=INDEX_NAME,
                                       revision=sha)
    if index_path.stat().st_size > MAX_INDEX_BYTES:
        raise TeacherAcquireError("Gemma index exceeds metadata safety limit")
    try:
        index = json.loads(index_path.read_text(encoding="utf8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherAcquireError("Gemma Safetensors index is malformed") from exc
    item = plan_from_index(index, layer, projection)
    shard_name = item["shard_filename"]
    candidates = [file for file in siblings if getattr(file, "rfilename", "") == shard_name]
    if len(candidates) != 1:
        raise TeacherAcquireError("Shard named by index is absent from the pinned official repository")
    expected_size = getattr(candidates[0], "size", None)
    if type(expected_size) is not int or expected_size <= 0:
        if download:
            raise TeacherAcquireError("Official shard download size is unavailable; refusing unknown-size download")
        expected_size = None
    if expected_size is not None and download and expected_size > max_shard_gib * 1024**3:
        raise TeacherAcquireError("Official model shard exceeds --max-shard-gib download cap")
    report = {
        "status": "TEACHER_SHARD_PLAN_ONLY",
        "source_repo": OFFICIAL_GEMMA_REPO,
        "official_license_url": OFFICIAL_GEMMA_PAGE,
        "pinned_revision": sha,
        "source_is_pretrained_verified": False,
        "shard_filename": shard_name, "shard_bytes": expected_size,
        "tensor_key": item["tensor_key"],
        "destination": str(root),
        "shard_path": str(root / shard_name),
        "index_path": str(index_path),
        "shard_downloaded": False, "whole_model_downloaded": False,
        "littlebit_model_trained": False,
        "ready_for_chat": False,
        "note": "The original licensed model requires authenticated access. "
                "Planning fetches only the index; --download fetches ONE shard.",
    }
    if not download:
        return report
    # Limit to one shard, plus disk headroom, even if the index references 12.
    free_bytes = shutil.disk_usage(root).free
    existing = root / shard_name
    if existing.is_symlink():
        raise TeacherAcquireError("Refusing a symlink pretrained shard path")
    if not existing.is_file():
        headroom = max(64 * 1024**2, min(1024**3, int(expected_size * 0.1)))
        if free_bytes < expected_size + headroom:
            raise TeacherAcquireError(
                "Not enough disk space for the original Gemma shard plus safe download headroom")
    actual = _download_within_root(hf_download, root=root, filename=shard_name,
                                   revision=sha)
    if actual.name != shard_name:
        raise TeacherAcquireError("Downloaded shard filename differs from pinned index")
    report["status"] = "TEACHER_SHARD_DOWNLOADED"
    report["shard_downloaded"] = True
    report["shard_path"] = str(actual)
    report["note"] = ("Downloaded only the requested original Gemma shard. "
                      "Now run LittleBitChat train layer/batch against this local directory. "
                      "No compressed full model has been trained.")
    return report
