"""Low-RAM exact Gemma 3 layer-0 attention-input reconstruction.

Reads only original token embedding rows and layer-0 RMSNorm weights from
local indexed Safetensors. This is valid ONLY for layer-0 q/k/v projection
inputs, before any attention or intermediate decoder block is evaluated.

No full model load, no invented activations, no remote files or implicit
downloads. Always binds to the exact separately loaded teacher matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .layer_train import _teacher_sha, load_projection


EMBED = "language_model.model.embed_tokens.weight"
NORM = "language_model.model.layers.0.input_layernorm.weight"
ATTN = "language_model.model.layers.0."
SUPPORTED = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
DEFAULT_EPS = 1e-6


class Layer0CaptureError(ValueError):
    pass


def _validate_target(layer, projection):
    if type(layer) is not int or layer != 0:
        raise Layer0CaptureError("Lightweight reconstruction supports layer 0 only")
    if projection not in SUPPORTED:
        raise Layer0CaptureError(
            "Layer-0 lightweight capture supports attention q_proj/k_proj/v_proj only")


def _source_index(checkpoint):
    root = Path(checkpoint)
    if not root.is_dir() or root.is_symlink():
        raise Layer0CaptureError("Expected a local original-model checkpoint directory")
    index = root / "model.safetensors.index.json"
    if not index.is_file() or index.is_symlink() or index.stat().st_size > 16*1024*1024:
        raise Layer0CaptureError("Missing or oversized Safetensors shard index")
    try:
        mapping = json.loads(index.read_text(encoding="utf8"))["weight_map"]
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise Layer0CaptureError("Malformed original checkpoint index") from exc
    if not isinstance(mapping, dict):
        raise Layer0CaptureError("Malformed weight_map in original checkpoint index")
    return root, mapping


def _shard(root, filename):
    if (not isinstance(filename, str) or not filename.endswith(".safetensors")
            or filename in (".", "..") or filename != Path(filename).name
            or "/" in filename or "\\" in filename):
        raise Layer0CaptureError("Unsafe checkpoint shard filename")
    path = root / filename
    if path.is_symlink():
        raise Layer0CaptureError("Unsafe symlink shard path")
    return path


def _inspect(root, projection, mapping):
    required=(EMBED,NORM,ATTN+projection+".weight")
    entries={}
    missing=[]
    shards_present=True
    for name in required:
        file=mapping.get(name)
        if file is None:
            missing.append(name)
            continue
        path=_shard(root,file)
        present=path.is_file()
        shards_present &= present
        entries[name]={"shard":file,"present":present}
    shapes={}
    for name, info in entries.items():
        if not info["present"]:
            continue
        path=root/info["shard"]
        try:
            with safe_open(str(path),framework="pt",device="cpu") as opened:
                if name not in opened.keys():
                    info["present"]=False
                    shards_present=False
                else:
                    shapes[name]=list(opened.get_slice(name).get_shape())
        except (OSError,RuntimeError,ValueError) as exc:
            info["present"]=False
            shards_present=False
            info["error"]=str(exc)[:250]
    return entries, missing, shards_present, shapes


def inspect_layer0_source(checkpoint, *, layer=0, projection="self_attn.q_proj"):
    """Metadata/header-only readiness report; never loads the embedding matrix."""
    _validate_target(layer,projection)
    root,mapping=_source_index(checkpoint)
    entries,missing,present,shapes=_inspect(root,projection,mapping)
    tokenizer=root/"tokenizer.json"
    config=root/"config.json"
    return {
        "status":"LAYER0_CAPTURE_PREFLIGHT",
        "source_layer":0,
        "projection":projection,
        "needed_weight_shards_present":bool(present and not missing),
        "missing_required_tensors":missing,
        "required_shards":entries,
        "embedding_shape":shapes.get(EMBED),
        "norm_shape":shapes.get(NORM),
        "projection_shape":shapes.get(ATTN+projection+".weight"),
        "tokenizer_json_present":bool(tokenizer.is_file() and not tokenizer.is_symlink()),
        "config_json_present":bool(config.is_file() and not config.is_symlink()),
        "ready_for_capture":bool(present and not missing and tokenizer.is_file()
                                  and not tokenizer.is_symlink()),
        "full_model_required":False,
        "source_provenance_independently_verified":False,
    }


def _read_config_eps(root, hidden):
    config=root/"config.json"
    if not config.is_file():
        return DEFAULT_EPS,False
    if config.is_symlink() or config.stat().st_size>1024*1024:
        raise Layer0CaptureError("Unsafe config.json")
    try:
        payload=json.loads(config.read_text(encoding="utf8"))
        text_config=payload.get("text_config",payload)
        # Gemma3TextConfig defaults rms_norm_eps to 1e-6 when omitted.
        # Preserve explicitly specified values for strict validation below.
        eps=text_config.get("rms_norm_eps", DEFAULT_EPS)
        dim=text_config.get("hidden_size")
    except (ValueError,AttributeError,TypeError) as exc:
        raise Layer0CaptureError("Invalid config.json for lightweight capture") from exc
    if type(dim) is not int or dim!=hidden:
        raise Layer0CaptureError("Config hidden_size disagrees with original embedding width")
    if type(eps) not in (int,float) or not math.isfinite(eps) or not 0<eps<=.001:
        raise Layer0CaptureError("Invalid original RMSNorm epsilon")
    return float(eps),True


@torch.no_grad()
def gemma3_layer0_qkv_inputs(token_ids, embedding_rows, norm, *, eps=DEFAULT_EPS):
    """Reproduce HF Gemma3ScaledEmbedding + Gemma3RMSNorm at BF16 precision.

    `embedding_rows` supports an nn-Embedding table tensor for testing or
    the specific token rows already sliced from Safetensors.
    """
    if type(eps) not in (float,int) or not math.isfinite(eps) or eps<=0:
        raise Layer0CaptureError("Invalid RMSNorm epsilon")
    if (not isinstance(norm,torch.Tensor) or norm.ndim!=1
            or not bool(torch.isfinite(norm).all())):
        raise Layer0CaptureError("Nonfinite or invalid original norm weights")
    if (not isinstance(token_ids,(list,tuple)) or not token_ids
            or any(type(x) is not int or x<0 for x in token_ids)):
        raise Layer0CaptureError("Invalid or missing original tokenizer IDs")
    if not isinstance(embedding_rows,torch.Tensor) or embedding_rows.ndim!=2:
        raise Layer0CaptureError("Expected BF16 original embedding tensor")
    if embedding_rows.shape[0]==len(token_ids):
        rows=embedding_rows
    else:
        if max(token_ids)>=embedding_rows.shape[0]:
            raise Layer0CaptureError("Token outside original embedding vocabulary")
        rows=embedding_rows[token_ids]
    if rows.shape[1]!=norm.shape[0] or not bool(torch.isfinite(rows).all()):
        raise Layer0CaptureError("Original embedding and norm widths disagree")
    # Gemma3 scaled embedding does multiplication in the embedding dtype;
    # gemma RMSNorm computes normalized float32 then casts to original dtype.
    scale=torch.tensor(math.sqrt(rows.shape[1]),dtype=rows.dtype,device=rows.device)
    embeddings=rows*scale
    float_embeddings=embeddings.float()
    normalized=float_embeddings*torch.rsqrt(
        float_embeddings.square().mean(-1,keepdim=True)+eps)
    return (normalized*(1.0+norm.float())).to(embeddings.dtype).float().contiguous()


def capture_layer0_from_original_weights(
        checkpoint, *, out, projection="self_attn.q_proj", layer=0,
        tokenizer, prompts, max_rows=512, max_tokens=128,
        allow_test_shapes=False, max_teacher_mib=512,
        token_disjoint=False, min_unique_tokens=64, token_split_seed=42):
    """Build a provenance-labeled 3-way activation bundle without full-model RAM.

    `tokenizer` must implement encode_conversation([('user', prompt)]).
    Safetensors get_slice fetches ONLY token embedding rows that are used.
    """
    _validate_target(layer,projection)
    if type(max_rows) is not int or not 16<=max_rows<=8192:
        raise Layer0CaptureError("max_rows must be 16..8192")
    if type(max_tokens) is not int or not 16<=max_tokens<=512:
        raise Layer0CaptureError("max_tokens must be 16..512")
    if type(allow_test_shapes) is not bool:
        raise Layer0CaptureError("allow_test_shapes must be boolean")
    if type(token_disjoint) is not bool:
        raise Layer0CaptureError("token_disjoint must be boolean")
    output=Path(out)
    if output.exists() or output.is_symlink():
        raise Layer0CaptureError("Capture file already exists or is unsafe")
    if set(prompts) != {"calibration","selection","audit"}:
        raise Layer0CaptureError("Require calibration/selection/audit prompt sets")
    all_prompts=[]
    for name,values in prompts.items():
        if (not isinstance(values,list) or not 3<=len(values)<=512
                or not all(isinstance(x,str) and x.strip() for x in values)
                or len(set(values)) != len(values)):
            raise Layer0CaptureError("Invalid or duplicate prompts in "+name)
        all_prompts.extend(values)
    if len(set(all_prompts)) != len(all_prompts):
        raise Layer0CaptureError("Capture prompt overlap or duplicate across splits")

    root,mapping=_source_index(checkpoint)
    _,missing,present,shapes=_inspect(root,projection,mapping)
    if missing or not present:
        raise Layer0CaptureError(
            "Original embedding/norm/projection shard missing; run layer0-preflight")
    embedding_shape=shapes[EMBED]
    norm_shape=shapes[NORM]
    projection_shape=shapes[ATTN+projection+".weight"]
    if (len(embedding_shape)!=2 or len(norm_shape)!=1 or
            len(projection_shape)!=2 or embedding_shape[1]!=norm_shape[0] or
            projection_shape[1]!=norm_shape[0]):
        raise Layer0CaptureError("Inconsistent original Gemma 3 source tensor shapes")
    if not allow_test_shapes and (
            embedding_shape != [262208,5376] or norm_shape != [5376] or
            (projection=="self_attn.q_proj" and projection_shape != [4096,5376])):
        raise Layer0CaptureError("Unexpected real Gemma 3 27B tensor shapes")

    token_plan=None
    if token_disjoint:
        # Plan the token-ID holdout before allocating the teacher matrix.
        from .token_disjoint import select_token_disjoint_ids, TokenDisjointError
        try:
            token_plan=select_token_disjoint_ids(
                tokenizer,prompts,max_tokens=max_tokens,max_rows=max_rows,
                min_unique_tokens=min_unique_tokens,
                vocab_size=embedding_shape[0],seed=token_split_seed)
        except TokenDisjointError as exc:
            raise Layer0CaptureError("Token-disjoint capture refused: "+str(exc)) from exc

    eps,config_verified=_read_config_eps(root,norm_shape[0])
    teacher=load_projection(root,layer=0,projection=projection,
                            max_teacher_mib=max_teacher_mib)
    # Bind to exact same teacher checkpoint consumed by activation-calibrate.
    if list(teacher.shape)!=projection_shape:
        raise Layer0CaptureError("Teacher projection shape mismatch")
    norm_path=_shard(root,mapping[NORM])
    embed_path=_shard(root,mapping[EMBED])
    with safe_open(str(norm_path),framework="pt",device="cpu") as opened:
        norm=opened.get_tensor(NORM).float()
    if not bool(torch.isfinite(norm).all()):
        raise Layer0CaptureError("Nonfinite original RMSNorm weights")

    import hashlib as _h
    hashes={name:_h.sha256(json.dumps(vals,ensure_ascii=False).encode("utf8")).hexdigest()
            for name,vals in prompts.items()}

    captured={}
    cache={}
    token_arrays={}
    with safe_open(str(embed_path),framework="pt",device="cpu") as opened:
        embedding_slice=opened.get_slice(EMBED)
        if list(embedding_slice.get_shape())!=embedding_shape:
            raise Layer0CaptureError("Original embedding shape changed")
        if token_plan is not None:
            # Exactly one sampled row per unique token ID; no repeated
            # special/prefix tokens or other token-ID overlap across splits.
            for name in ("calibration","selection","audit"):
                ids=token_plan["ids"][name]
                for idx in ids:
                    if idx not in cache:
                        cache[idx]=embedding_slice[idx:idx+1].clone()[0]
                rows=torch.stack([cache[idx] for idx in ids],dim=0)
                captured[name]=gemma3_layer0_qkv_inputs(
                    ids,rows,norm,eps=eps).float().contiguous()
                token_arrays["token_ids_"+name]=torch.tensor(ids,dtype=torch.int64)
        else:
            # Preserve the original prompt-disjoint format for compatibility.
            for name in ("calibration","selection","audit"):
                collected=[]
                count=0
                for prompt in prompts[name]:
                    if count>=max_rows:
                        break
                    ids=list(tokenizer.encode_conversation([("user",prompt)]))[:max_tokens]
                    if not ids or any(type(x) is not int or not 0<=x<embedding_shape[0] for x in ids):
                        raise Layer0CaptureError("Tokenizer produced invalid embedding token IDs")
                    for idx in set(ids):
                        if idx not in cache:
                            cache[idx]=embedding_slice[idx:idx+1].clone()[0]
                    rows=torch.stack([cache[idx] for idx in ids[:max_rows-count]],dim=0)
                    capture=gemma3_layer0_qkv_inputs(ids[:len(rows)],rows,norm,eps=eps)
                    collected.append(capture)
                    count+=len(capture)
                if count<16:
                    raise Layer0CaptureError(name+" contains fewer than 16 captured activations")
                captured[name]=torch.cat(collected,dim=0).float().contiguous()

    from .activation_calibration import _sample_hashes
    samples={name:_sample_hashes(values) for name,values in captured.items()}
    for left,right in (("calibration","selection"),("calibration","audit"),
                       ("selection","audit")):
        if token_plan is not None:
            if samples[left] & samples[right]:
                raise Layer0CaptureError("Duplicate activation rows across token-disjoint splits")
        elif samples[left].issubset(samples[right]) or samples[right].issubset(samples[left]):
            raise Layer0CaptureError("Whole activation split overlap: "+left+"/"+right)
    meta={
        "teacher_sha256":_teacher_sha(teacher),
        "layer":"0",
        "projection":projection,
        "input_module":ATTN+projection,
        "capture_method":"gemma3_layer0_embedding_rmsnorm",
        "split_method":"disjoint_token_ids_v1" if token_plan is not None else "disjoint_prompts",
        "source_model":str(root.resolve()),
        "rms_norm_eps":str(eps),
        "config_present_and_checked":str(config_verified).lower(),
        "embedding_tensor":EMBED,
        "normalization_tensor":NORM,
        "calibration_text_sha256":hashes["calibration"],
        "selection_text_sha256":hashes["selection"],
        "audit_text_sha256":hashes["audit"],
    }
    if token_plan is not None:
        meta.update({
            "token_id_selection_algorithm":token_plan["selection_algorithm"],
            "token_split_seed":str(token_split_seed),
            "min_unique_tokens":str(min_unique_tokens),
            "shared_token_ids_dropped":str(token_plan["shared_dropped"]),
        })
    output.parent.mkdir(parents=True,exist_ok=True)
    save_file({**captured, **token_arrays},str(output),metadata=meta)
    report={
        "status":"LAYER0_TOKEN_DISJOINT_CAPTURE" if token_plan is not None
                 else "LAYER0_EXACT_EMBED_NORM_CAPTURE",
        "file":str(output),
        "layer":0,
        "projection":projection,
        "teacher_sha256":meta["teacher_sha256"],
        "capture_method":meta["capture_method"],
        "calibration_rows":len(captured["calibration"]),
        "selection_rows":len(captured["selection"]),
        "audit_rows":len(captured["audit"]),
        "unique_embedding_rows_read":len(cache),
        "full_model_required":False,
        "config_present_and_checked":config_verified,
        "provenance_independently_verified":False,
        "training_was_run":False,
        "token_disjoint_verified":bool(token_plan is not None),
    }
    if token_plan is not None:
        report.update({
            "shared_token_ids_dropped":token_plan["shared_dropped"],
            "source_unique_token_counts":token_plan["source_unique_counts"],
            "eligible_unique_token_counts":token_plan["eligible_unique_counts"],
            "unique_token_counts":token_plan["sampled_unique_counts"],
            "min_unique_tokens":min_unique_tokens,
            "token_id_selection_algorithm":token_plan["selection_algorithm"],
        })
    return report
