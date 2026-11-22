"""Explicit, no-download PyTorch forward-pre-hook capture of projection inputs.

The caller supplies an already-loaded original model and disjoint batches.
The observed nn.Linear weight must equal the local teacher projection bytes.
This module does not load 27B models or assert model/source authenticity.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from .activation_calibration import _valid_sha
from .layer_train import _teacher_sha


class _CaptureComplete(Exception):
    """Private signal to stop computation after observing target projection."""


@torch.no_grad()
def capture_projection_inputs(
        model, teacher, *, calibration_batches, selection_batches, audit_batches,
        out, layer, projection, source_model, calibration_text_sha256,
        selection_text_sha256, audit_text_sha256, max_rows=512,
        stop_after_hook=True):
    """Capture exact inputs entering a selected linear projection via a pre-hook.

    Batches can be argument tensors (model(batch)) or mappings of keyword
    tensors (model(**batch)). Stop the remaining forward immediately after
    reaching the pre-hook by default to avoid expensive decoder logits.
    The module must be active during each forward;
    the capture fails rather than emitting fabricated data if the hook
    never fires. Runtime/VRAM requirements are dictated by caller's model.
    """
    if type(max_rows) is not int or not 16 <= max_rows <= 8192:
        raise ValueError("max_rows must be 16..8192")
    if type(layer) is not int or not 0 <= layer < 62:
        raise ValueError("Invalid target layer")
    if not isinstance(projection, str) or not projection:
        raise ValueError("Invalid projection name")
    if not source_model or not isinstance(source_model, str):
        raise ValueError("Source model description required")
    groups = [calibration_text_sha256, selection_text_sha256, audit_text_sha256]
    if not all(_valid_sha(value) for value in groups) or len(set(groups)) != 3:
        raise ValueError("Three distinct prompt-split SHA-256 hashes required")
    module_name=f"language_model.model.layers.{layer}.{projection}"
    named=dict(model.named_modules())
    module=named.get(module_name)
    if module is None or not isinstance(module, torch.nn.Linear):
        raise ValueError("Original model's target linear module missing: "+module_name)
    if not isinstance(teacher, torch.Tensor) or teacher.ndim != 2:
        raise ValueError("Invalid teacher weight")
    if (tuple(module.weight.shape) != tuple(teacher.shape)
            or _teacher_sha(module.weight) != _teacher_sha(teacher)):
        raise ValueError("Original model module weight differs from supplied teacher")
    if model.training or module.training:
        raise ValueError("Capture requires a model in eval mode")
    if Path(out).exists() or Path(out).is_symlink():
        raise ValueError("Capture output exists or is unsafe")

    samples={}
    for name, batches in (("calibration", calibration_batches),
                          ("selection", selection_batches),
                          ("audit", audit_batches)):
        rows=[]
        count=0
        def hook(_, args):
            nonlocal count
            if not args or not isinstance(args[0],torch.Tensor):
                raise ValueError("Expected tensor input to captured linear projection")
            x=args[0].detach().to("cpu",dtype=torch.float32).reshape(-1,teacher.shape[1])
            if not bool(torch.isfinite(x).all()):
                raise ValueError("Nonfinite captured projection inputs")
            if count < max_rows:
                take=min(max_rows-count,x.shape[0])
                rows.append(x[:take].clone())
                count+=take
            if stop_after_hook:
                raise _CaptureComplete()
        handle=module.register_forward_pre_hook(hook)
        try:
            for batch in batches:
                if count >= max_rows:
                    break
                try:
                    if isinstance(batch,dict):
                        model(**batch)
                    else:
                        model(batch)
                except _CaptureComplete:
                    pass
        finally:
            handle.remove()
        if count < 16:
            raise ValueError(name+" forward pre-hook captured fewer than 16 tokens")
        samples[name]=torch.cat(rows,dim=0).contiguous()

    from .activation_calibration import _sample_hashes
    hashes={name:_sample_hashes(x) for name,x in samples.items()}
    # Shared prefix tokens can naturally repeat across disjoint prompt sets,
    # but a fully reused split is not a valid independent evaluation.
    for left,right in (("calibration","selection"),("calibration","audit"),
                       ("selection","audit")):
        if hashes[left].issubset(hashes[right]) or hashes[right].issubset(hashes[left]):
            raise ValueError("Whole activation split overlap: "+left+"/"+right)
    metadata={
        "teacher_sha256":_teacher_sha(teacher),
        "layer":str(layer),
        "projection":projection,
        "input_module":module_name,
        "capture_method":"forward_pre_hook",
        "split_method":"disjoint_prompts",
        "source_model":source_model,
        "calibration_text_sha256":calibration_text_sha256,
        "selection_text_sha256":selection_text_sha256,
        "audit_text_sha256":audit_text_sha256,
    }
    output=Path(out)
    output.parent.mkdir(parents=True,exist_ok=True)
    save_file(samples,str(output),metadata=metadata)
    return {
        "status":"PROJECTION_INPUTS_CAPTURED",
        "file":str(output),
        "teacher_sha256":metadata["teacher_sha256"],
        "calibration_rows":len(samples["calibration"]),
        "selection_rows":len(samples["selection"]),
        "audit_rows":len(samples["audit"]),
        "capture_method":"forward_pre_hook",
        "provenance_independently_verified":False,
        "trained_full_model":False,
    }
