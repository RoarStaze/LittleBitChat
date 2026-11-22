"""Explicit local-only capture from original Google Gemma 3 on adequate RAM.

Heavy dependencies are imported only after the user confirms that loading
the complete ORIGINAL model into host RAM is acceptable. No downloads,
inferred activations, remote calls, or implicit CUDA allocations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .activation_capture import capture_projection_inputs
from .layer_train import TrainingError, load_projection


def _read_prompts(path):
    source=Path(path)
    if (not source.is_file() or source.is_symlink()
            or source.stat().st_size > 2 * 1024 * 1024):
        raise TrainingError("Missing, unsafe, or oversized prompt file: "+str(source))
    prompts=[line.strip() for line in source.read_text(encoding="utf8").splitlines()
             if line.strip()]
    if not 3 <= len(prompts) <= 512:
        raise TrainingError("Each prompt split requires 3..512 nonempty lines")
    if len(set(prompts)) != len(prompts):
        raise TrainingError("Duplicate prompts within a capture split")
    return prompts


def run_capture(args):
    if not args.confirm_full_model_memory:
        raise TrainingError(
            "Refusing original 27B model load without --confirm-full-model-memory; "
            "the uncompressed model requires substantial host RAM")
    checkpoint=Path(args.checkpoint)
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise TrainingError("Capture requires a complete local original-model directory")
    if type(args.max_tokens) is not int or not 16 <= args.max_tokens <= 512:
        raise TrainingError("max-tokens must be in 16..512")
    if type(args.max_rows) is not int or not 16 <= args.max_rows <= 8192:
        raise TrainingError("max-rows must be in 16..8192")
    groups = {
        "calibration":_read_prompts(args.calibration_texts),
        "selection":_read_prompts(args.selection_texts),
        "audit":_read_prompts(args.audit_texts),
    }
    for left,right in (("calibration","selection"),("calibration","audit"),
                       ("selection","audit")):
        if set(groups[left]) & set(groups[right]):
            raise TrainingError("Capture prompt split overlap: "+left+"/"+right)
    hashes={
        key:hashlib.sha256(json.dumps(value,sort_keys=False,
                      ensure_ascii=False).encode("utf8")).hexdigest()
        for key,value in groups.items()
    }
    teacher=load_projection(checkpoint,layer=args.layer,projection=args.projection,
                            max_teacher_mib=args.max_teacher_mib)
    try:
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    except ImportError as exc:
        raise TrainingError(
            "Capture requires optional 'transformers' and 'accelerate' packages "
            "in the capturing environment") from exc
    try:
        # Local-only, no implicit fetching, BF16 CPU to avoid VRAM overcommit.
        model=Gemma3ForConditionalGeneration.from_pretrained(
            str(checkpoint), local_files_only=True, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map="cpu")
        tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),
                                                 local_files_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrainingError(
            "Unable to load the complete original Gemma 3 checkpoint locally; "
            "use a machine with adequate RAM: "+str(exc)) from exc
    model.eval()

    def batches(texts):
        for text in texts:
            yield dict(tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=args.max_tokens))

    return capture_projection_inputs(
        model,teacher,
        calibration_batches=batches(groups["calibration"]),
        selection_batches=batches(groups["selection"]),
        audit_batches=batches(groups["audit"]),
        out=args.out, layer=args.layer, projection=args.projection,
        source_model=str(checkpoint.resolve()),
        calibration_text_sha256=hashes["calibration"],
        selection_text_sha256=hashes["selection"],
        audit_text_sha256=hashes["audit"], max_rows=args.max_rows,
        stop_after_hook=True)
