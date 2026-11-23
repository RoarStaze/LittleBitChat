"""Large-scale Gemma layer-0 token-type capture with a permanently reused audit lock.

Uniformly sample original tokenizer vocabulary *types*, not natural-language
token frequencies. Only local indexed original Gemma embedding/norm/projection
weights are read. Neither a full 27B model nor network access is required.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .layer0_capture import (
    EMBED, NORM, ATTN, SUPPORTED, _source_index, _inspect, _shard,
    _validate_target, _read_config_eps, gemma3_layer0_qkv_inputs,
)
from .layer_train import _teacher_sha, load_projection


class FrozenAuditError(ValueError):
    pass


SPLITS = ("calibration", "selection", "audit")
LOCK_VERSION = "FROZEN_LAYER0_TOKEN_AUDIT_V1"
SAMPLER = "sha256_uniform_unique_vocabulary_ids_v1"
MAX_LOCK_BYTES = 256 * 1024


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),
                       ensure_ascii=False)+"\n").encode("utf8")


def _tensor_sha(x):
    return _sha(x.contiguous().numpy().tobytes())


def _rank_key(token, seed, purpose):
    return hashlib.sha256(
        f"littlebit-frozen-v1:{purpose}:{seed}:{token}".encode("ascii")).digest()


def _source_token_pool(tokenizer, *, vocab_size, source_text_file=None):
    """Return valid, normal vocabulary IDs and a tokenizer/source digest."""
    backend=getattr(tokenizer,"backend",None)
    if (backend is None or not hasattr(backend,"id_to_token")
            or getattr(tokenizer,"vocab_size",None) != vocab_size):
        raise FrozenAuditError(
            "Matching original Gemma tokenizer with full embedding-sized vocabulary required")
    h=hashlib.sha256()
    valid=set()
    for tid in range(vocab_size):
        token=backend.id_to_token(tid)
        if not isinstance(token,str) or not token:
            continue
        h.update(str(tid).encode("ascii")+b"\0"+token.encode("utf8")+b"\n")
        # Exclude control, image and reserved tokenizer IDs; this is a
        # text-token TYPE study, not a multimodal or control-token test.
        if token.startswith("<") and token.endswith(">"):
            continue
        valid.add(tid)
    tokenizer_digest=h.hexdigest()
    if source_text_file is None:
        return valid, tokenizer_digest, "uniform_vocabulary_types", None
    path=Path(source_text_file)
    if (not path.is_file() or path.is_symlink()
            or not 1<=path.stat().st_size<=32*1024*1024):
        raise FrozenAuditError("Missing, unsafe or oversized local source-text file")
    if not hasattr(tokenizer,"_encode"):
        raise FrozenAuditError("Tokenizer does not expose original text-only encode method")
    raw=path.read_bytes()
    try:
        content=raw.decode("utf8")
    except UnicodeDecodeError as exc:
        raise FrozenAuditError("Source text must be valid UTF-8") from exc
    if not content.strip():
        raise FrozenAuditError("Source text file must contain substantive UTF-8 text")
    ids=set()
    for line in content.splitlines():
        if not line.strip():
            continue
        # Bound per-call tokenizer memory; a long paragraph can still
        # contribute tokens through consecutive 4k-character windows.
        for begin in range(0,len(line),4096):
            ids.update(tokenizer._encode(line[begin:begin+4096]))
    return valid.intersection(ids),tokenizer_digest,"local_text_unique_types",_sha(raw)


def _plan_ids(pool, *, calibration_tokens, selection_tokens, audit_tokens,
              sample_seed, audit_seed, frozen_ids=None):
    for name,count in (("calibration_tokens",calibration_tokens),
                       ("selection_tokens",selection_tokens),
                       ("audit_tokens",audit_tokens)):
        if type(count) is not int or not 16<=count<=8192:
            raise FrozenAuditError(f"{name} must be within 16..8192")
    for name,seed in (("sample_seed",sample_seed),("audit_seed",audit_seed)):
        if type(seed) is not int or not 0<=seed<2**31:
            raise FrozenAuditError(f"{name} must be a nonnegative integer below 2**31")
    if frozen_ids is None:
        if len(pool)<calibration_tokens+selection_tokens+audit_tokens:
            raise FrozenAuditError(
                f"Insufficient eligible unique vocabulary IDs: {len(pool)}; "
                f"need {calibration_tokens+selection_tokens+audit_tokens}")
        aud=sorted(pool,key=lambda t:_rank_key(t,audit_seed,"audit"))[:audit_tokens]
    else:
        aud=list(frozen_ids)
        if (len(aud)!=audit_tokens or len(set(aud))!=len(aud)
                or any(type(x) is not int or x<0 for x in aud)):
            raise FrozenAuditError("Frozen audit lock has invalid token ID count or contents")
    remaining=pool.difference(aud)
    if len(remaining)<calibration_tokens+selection_tokens:
        raise FrozenAuditError(
            f"Insufficient eligible unique training/selection vocabulary IDs: "
            f"{len(remaining)} after excluding locked audit")
    ordered=sorted(remaining,key=lambda t:_rank_key(t,sample_seed,"fit"))
    ids={
        "calibration":ordered[:calibration_tokens],
        "selection":ordered[calibration_tokens:calibration_tokens+selection_tokens],
        "audit":aud,
    }
    if any(set(ids[a])&set(ids[b]) for a,b in (
            ("calibration","selection"),("calibration","audit"),
            ("selection","audit"))):
        raise FrozenAuditError("Internal error: frozen audit token ID leakage")
    return ids


def _load_lock(path):
    target=Path(path)
    if (not target.is_file() or target.is_symlink()
            or not 1<=target.stat().st_size<=MAX_LOCK_BYTES):
        raise FrozenAuditError("Frozen audit lock missing, unsafe or oversized")
    raw=target.read_bytes()
    try:
        lock=json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError,ValueError) as exc:
        raise FrozenAuditError("Malformed frozen audit lock JSON") from exc
    if not isinstance(lock,dict) or raw!=_canonical(lock):
        raise FrozenAuditError("Frozen audit lock must remain canonical and unmodified")
    if lock.get("lock_version")!=LOCK_VERSION:
        raise FrozenAuditError("Unsupported frozen audit lock version")
    return lock,_sha(raw)


def verify_frozen_audit_lock(bundle, audit_lock, teacher, *, layer, projection,
                             expected_sha256=None):
    """Validate lock byte digest, provenance, ID labels and exact audit rows.

    The reference is frozen by an on-disk exclusive-create lock. The caller
    should also retain its printed SHA-256 independently; simultaneous
    malicious edits to both data and lock cannot be detected without an
    external trusted anchor.
    """
    lock, lock_digest=_load_lock(audit_lock)
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256,str) or len(expected_sha256)!=64
                or any(c not in "0123456789abcdef" for c in expected_sha256)
                or expected_sha256!=lock_digest):
            raise FrozenAuditError("Frozen audit external expected SHA-256 fingerprint mismatch")
    if not getattr(bundle,"token_disjoint_verified",False) or bundle.token_ids is None:
        raise FrozenAuditError("Frozen audit requires strictly token-disjoint inputs")
    if (lock.get("teacher_sha256")!=_teacher_sha(teacher)
            or lock.get("layer")!=layer or lock.get("projection")!=projection
            or bundle.metadata.get("teacher_sha256")!=lock.get("teacher_sha256")):
        raise FrozenAuditError("Frozen audit lock teacher/layer/projection mismatch")
    if bundle.metadata.get("frozen_audit_lock_sha256")!=lock_digest:
        raise FrozenAuditError("Frozen audit lock SHA-256 mismatch")
    if bundle.metadata.get("tokenizer_sha256")!=lock.get("tokenizer_sha256"):
        raise FrozenAuditError("Frozen audit tokenizer binding mismatch")
    if (bundle.metadata.get("source_sampling")!=lock.get("source_sampling")
            or bundle.metadata.get("audit_seed")!=str(lock.get("audit_seed"))):
        raise FrozenAuditError("Frozen audit source/sampling parameters mismatch")
    ids=bundle.token_ids["audit"].tolist()
    if (ids!=lock.get("audit_token_ids")
            or len(ids)!=lock.get("audit_count")
            or _tensor_sha(bundle.audit)!=lock.get("audit_activation_sha256")
            or _sha(_canonical(ids))!=lock.get("audit_token_ids_sha256")):
        raise FrozenAuditError("Frozen audit token IDs or activation rows mismatch")
    return {
        "frozen_audit_verified":True,
        "frozen_audit_lock_sha256":lock_digest,
        "frozen_audit_rows":len(ids),
        "frozen_audit_token_ids_sha256":lock["audit_token_ids_sha256"],
        "frozen_audit_activation_sha256":lock["audit_activation_sha256"],
        "frozen_audit_sampling":lock["source_sampling"],
        "frozen_audit_provenance_independently_verified":False,
    }


@torch.no_grad()
def capture_vocabulary_checkpoint(
        checkpoint, *, out, audit_lock, tokenizer,
        layer=0, projection="self_attn.q_proj",
        calibration_tokens=2048, selection_tokens=1024, audit_tokens=1024,
        sample_seed=42, audit_seed=20261122, source_text_file=None,
        allow_test_shapes=False, max_teacher_mib=512,
        expected_audit_lock_sha256=None):
    """Create a LARGE, disjoint, exact original-source vocabulary capture.

    First run freezes the audit lock with exclusive-create, never overwrites.
    Later runs reuse exactly its audit IDs and exact activation content,
    allowing calibration/selection seed changes without audit drift.
    """
    _validate_target(layer,projection)
    out=Path(out)
    lock_path=Path(audit_lock)
    if out.exists() or out.is_symlink():
        raise FrozenAuditError("Capture output already exists; choose a fresh path")
    if lock_path.is_symlink() or not lock_path.parent.exists():
        raise FrozenAuditError("Unsafe or missing frozen-audit lock parent directory")
    if out.resolve()==lock_path.resolve():
        raise FrozenAuditError("Frozen audit lock path and capture output must differ")
    root,mapping=_source_index(checkpoint)
    _,missing,present,shapes=_inspect(root,projection,mapping)
    if missing or not present:
        raise FrozenAuditError("Original embedding/norm/projection shard missing")
    e_shape=shapes[EMBED]
    n_shape=shapes[NORM]
    p_shape=shapes[ATTN+projection+".weight"]
    if (len(e_shape)!=2 or len(n_shape)!=1 or len(p_shape)!=2
            or e_shape[1]!=n_shape[0] or p_shape[1]!=n_shape[0]):
        raise FrozenAuditError("Inconsistent source original model tensor dimensions")
    if not allow_test_shapes and (
            e_shape!=[262208,5376] or n_shape!=[5376]
            or (projection=="self_attn.q_proj" and p_shape!=[4096,5376])):
        raise FrozenAuditError("Unexpected original Gemma3-27B layer-0 dimensions")
    # Avoid writing an artifact that the default 256MiB activation loader
    # cannot subsequently load without an explicit limit override.
    approx_bytes=sum((calibration_tokens,selection_tokens,audit_tokens))*e_shape[1]*4
    if approx_bytes>250*1024*1024:
        raise FrozenAuditError(
            "Capture would exceed default activation loader limit; reduce token counts")
    pool,tokenizer_sha,source_label,source_sha=_source_token_pool(
        tokenizer,vocab_size=e_shape[0],source_text_file=source_text_file)
    existing=None
    if lock_path.exists():
        existing,_=_load_lock(lock_path)
        if (existing.get("tokenizer_sha256")!=tokenizer_sha
                or existing.get("audit_seed")!=audit_seed
                or existing.get("audit_count")!=audit_tokens
                or existing.get("source_sampling")!=source_label):
            raise FrozenAuditError(
                "Existing frozen audit lock parameters disagree; do not rewrite it")
    ids=_plan_ids(
        pool,calibration_tokens=calibration_tokens,
        selection_tokens=selection_tokens,audit_tokens=audit_tokens,
        sample_seed=sample_seed,audit_seed=audit_seed,
        frozen_ids=existing.get("audit_token_ids") if existing else None)
    eps,config_checked=_read_config_eps(root,e_shape[1])
    teacher=load_projection(root,layer=0,projection=projection,
                            max_teacher_mib=max_teacher_mib)
    if list(teacher.shape)!=p_shape:
        raise FrozenAuditError("Original projection changed shape")
    teacher_sha=_teacher_sha(teacher)
    if existing is not None and (
            existing.get("teacher_sha256")!=teacher_sha
            or existing.get("layer")!=0
            or existing.get("projection")!=projection):
        raise FrozenAuditError("Existing frozen audit lock teacher binding mismatch")
    with safe_open(str(_shard(root,mapping[NORM])),framework="pt",device="cpu") as reader:
        norm=reader.get_tensor(NORM).float()
    if not bool(torch.isfinite(norm).all()):
        raise FrozenAuditError("Nonfinite original Gemma RMSNorm")
    captures={}
    token_arrays={}
    with safe_open(str(_shard(root,mapping[EMBED])),framework="pt",device="cpu") as reader:
        embed=reader.get_slice(EMBED)
        if list(embed.get_shape())!=e_shape:
            raise FrozenAuditError("Original embedding shard changed during capture")
        for name in SPLITS:
            token_ids=ids[name]
            # Read only needed original embedding rows; no full vocabulary
            # matrix is ever materialized into host RAM.
            original_rows=torch.cat(
                [embed[tid:tid+1].clone() for tid in token_ids],dim=0)
            captures[name]=gemma3_layer0_qkv_inputs(
                token_ids,original_rows,norm,eps=eps)
            token_arrays["token_ids_"+name]=torch.tensor(token_ids,dtype=torch.int64)
    from .activation_calibration import _sample_hashes
    hashes={name:_sample_hashes(captures[name]) for name in SPLITS}
    for a,b in (("calibration","selection"),("calibration","audit"),
                ("selection","audit")):
        if hashes[a]&hashes[b]:
            raise FrozenAuditError("Original model has duplicate activation bytes across splits")
    lock={
        "lock_version":LOCK_VERSION,
        "teacher_sha256":teacher_sha,
        "layer":0,
        "projection":projection,
        "tokenizer_sha256":tokenizer_sha,
        "audit_seed":audit_seed,
        "audit_count":audit_tokens,
        "source_sampling":source_label,
        "audit_token_ids":ids["audit"],
        "audit_token_ids_sha256":_sha(_canonical(ids["audit"])),
        "audit_activation_sha256":_tensor_sha(captures["audit"]),
    }
    if existing is not None and existing!=lock:
        raise FrozenAuditError(
            "Existing frozen audit lock differs from reconstructed audit; refusing drift")
    lock_raw=_canonical(lock)
    lock_digest=_sha(lock_raw)
    if expected_audit_lock_sha256 is not None:
        if (not isinstance(expected_audit_lock_sha256,str)
                or len(expected_audit_lock_sha256)!=64
                or any(c not in "0123456789abcdef"
                       for c in expected_audit_lock_sha256)
                or expected_audit_lock_sha256!=lock_digest):
            raise FrozenAuditError(
                "Frozen audit external expected SHA-256 fingerprint mismatch")
    meta={
        "teacher_sha256":teacher_sha,
        "layer":"0",
        "projection":projection,
        "input_module":ATTN+projection,
        "capture_method":"gemma3_layer0_embedding_rmsnorm",
        "split_method":"disjoint_token_ids_v1",
        "source_model":str(root.resolve()),
        "rms_norm_eps":str(eps),
        "config_present_and_checked":str(config_checked).lower(),
        "embedding_tensor":EMBED,
        "normalization_tensor":NORM,
        "calibration_text_sha256":_sha(_canonical(ids["calibration"])),
        "selection_text_sha256":_sha(_canonical(ids["selection"])),
        "audit_text_sha256":_sha(_canonical(ids["audit"])),
        "source_prompt_kind":"tokenizer_vocabulary_ids_not_prompt_texts",
        "source_sampling":source_label,
        "source_text_sha256":source_sha or "not_applicable",
        "tokenizer_sha256":tokenizer_sha,
        "audit_seed":str(audit_seed),
        "sample_seed":str(sample_seed),
        "frozen_audit_lock_sha256":lock_digest,
        "token_id_selection_algorithm":SAMPLER,
        "min_unique_tokens":str(min(calibration_tokens,selection_tokens,audit_tokens)),
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() or out.is_symlink():
        raise FrozenAuditError("Capture output appeared while preparing data")
    save_file({**captures,**token_arrays},str(out),metadata=meta)
    new_lock=False
    try:
        if existing is None:
            with lock_path.open("xb") as writer:
                writer.write(lock_raw)
            new_lock=True
        from .activation_calibration import load_activation_bundle
        from .layer0_capture import verify_token_disjoint_source
        bundle=load_activation_bundle(
            out,teacher,layer=0,projection=projection,require_token_disjoint=True)
        frozen=verify_frozen_audit_lock(
            bundle,lock_path,teacher,layer=0,projection=projection,
            expected_sha256=expected_audit_lock_sha256)
        replay=verify_token_disjoint_source(
            root,bundle,layer=0,projection=projection)
    except Exception:
        out.unlink(missing_ok=True)
        if new_lock:
            lock_path.unlink(missing_ok=True)
        raise
    return {
        "status":"FROZEN_TOKEN_AUDIT_CAPTURE",
        "file":str(out),
        "frozen_audit_lock":str(lock_path),
        "frozen_audit_lock_created":new_lock,
        "teacher_sha256":teacher_sha,
        "token_disjoint_verified":True,
        "split_method":"disjoint_token_ids_v1",
        "unique_token_counts":{k:len(v) for k,v in ids.items()},
        "eligible_token_type_count":len(pool),
        "source_sampling":source_label,
        "source_text_sha256":source_sha,
        "full_model_required":False,
        "training_was_run":False,
        "validated_language_quality":False,
        **frozen,**replay,
    }


def paired_audit_bootstrap(before_errors, after_errors, *,
                           resamples=1000, seed=20261122):
    """Paired token-level bootstrap of relative NRMSE improvement.

    Each observation is ONE disjoint token-ID row. Bootstrap uncertainty
    applies to this sampled token distribution, not actual language quality.
    Audit must never be used by the optimizer or acceptance gate.
    """
    if type(resamples) is not int or not 100<=resamples<=10000:
        raise FrozenAuditError("Bootstrap resamples must be 100..10000")
    if type(seed) is not int or not 0<=seed<2**31:
        raise FrozenAuditError("Invalid bootstrap seed")
    if (not isinstance(before_errors,torch.Tensor)
            or not isinstance(after_errors,torch.Tensor)
            or before_errors.ndim!=2
            or before_errors.shape!=after_errors.shape
            or before_errors.shape[0]<16
            or not bool(torch.isfinite(before_errors).all())
            or not bool(torch.isfinite(after_errors).all())):
        raise FrozenAuditError("Paired bootstrap requires finite aligned audit error rows")
    a=before_errors.detach().float().square().mean(1).cpu().numpy().astype("float64")
    b=after_errors.detach().float().square().mean(1).cpu().numpy().astype("float64")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or float(a.sum())<=0:
        raise FrozenAuditError("Invalid paired audit baseline error")
    rng=np.random.default_rng(seed)
    vals=np.empty(resamples,dtype="float64")
    for k in range(resamples):
        sample=rng.integers(0,len(a),size=len(a))
        baseline=float(a[sample].sum())
        corrected=float(b[sample].sum())
        vals[k]=100*(1-math.sqrt(corrected/baseline)) if baseline>0 else float("nan")
    if not np.isfinite(vals).all():
        raise FrozenAuditError("Nonfinite paired bootstrap estimates")
    return {
        "audit_relative_nrmse_improvement_pct":100*(1-math.sqrt(float(b.sum()/a.sum()))),
        "audit_relative_improvement_ci95_pct":[float(v) for v in np.quantile(vals,[.025,.975])],
        "audit_bootstrap_resamples":resamples,
        "audit_bootstrap_seed":seed,
        "audit_bootstrap_unit":"unique_frozen_token_id",
        "audit_bootstrap_distribution_warning":
            "Token-type bootstrap; not representative of natural token frequencies "
            "or end-to-end language quality. Audit is descriptive, not a selection gate.",
    }
