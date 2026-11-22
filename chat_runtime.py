"""Out-of-core Gemma 3 LittleBit chat runtime.

This executes real-format packed Gemma3 projections, NOT fake chat text.
A valid *trained* 27B LittleBit-2 checkpoint is still required to produce
meaningful replies. Existing public dense Gemma checkpoints are rejected.

Every forward token streams 62 decoder blocks sequentially. Embeddings are
read in bounded disk slices; 62 FP32 K/V states persist in a capped CuPy
pool. GPU speed, numerical correctness at real 27B size and memory fit are
NOT yet hardware-verified. User consent is required for unverified weights.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import time
from types import SimpleNamespace

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from gemma3_bridge import read_gemma_config, test_fixture_path, Gemma3TextArchitecture
from gemma3_27b_ready import (LargeGemmaCheckpoint, MemoryLimits, MIB,
                               build_report, CheckpointRejected)
from safetensors_layer import _ShardGroup, CheckpointFormatError, DTYPES

EMBED = "language_model.model.embed_tokens.weight"
FINAL_NORM = "language_model.model.norm.weight"
NORM_SUFFIXES = ("input_layernorm", "post_attention_layernorm",
                 "pre_feedforward_layernorm", "post_feedforward_layernorm",
                 "self_attn.q_norm", "self_attn.k_norm")


class ChatUnavailable(RuntimeError):
    """Required real weights, tokenizer, or hardware are unavailable."""


class CheckpointEmbeddings:
    """Read one token row or bounded output-head chunk; never full vocabulary.

    Gemma 3 ties the LM head to input token embeddings. This uses direct
    offsets from validated Safetensors descriptors. BF16 is converted in
    small batches, preserving the tied-head semantics without 2.6GB FP32
    copies. Disk streaming each generated token can be extremely slow.
    """

    def __init__(self, group: _ShardGroup, *, expected_vocab: int, expected_hidden: int):
        if not isinstance(group, _ShardGroup):
            raise TypeError("Expected validated Safetensors shard group")
        info = group.descriptor(EMBED)
        if tuple(info["shape"]) != (expected_vocab, expected_hidden):
            raise ChatUnavailable("Checkpoint's embedding shape does not match Gemma configuration")
        if info["dtype"] not in ("F32", "BF16", "F16"):
            raise ChatUnavailable("Unsupported text embedding dtype")
        self.group = group
        self.dtype = info["dtype"]
        self.vocab = expected_vocab
        self.hidden = expected_hidden
        self.bytes_per_element = DTYPES[self.dtype][1]
        self.base = info["data_offsets"][0]
        self.shard = group._get_shard(EMBED)

    def rows(self, start: int, count: int):
        if (type(start) is not int or type(count) is not int or
                start < 0 or not 1 <= count <= 2048 or start+count > self.vocab):
            raise ValueError("Embedding chunk outside vocabulary or chunk budget")
        byte_count = count * self.hidden * self.bytes_per_element
        absolute = self.shard.data_start + self.base + start*self.hidden*self.bytes_per_element
        with self.shard.path.open("rb") as f:
            f.seek(absolute)
            data = f.read(byte_count)
        if len(data) != byte_count:
            raise ChatUnavailable("Short embedding read; checkpoint changed during execution")
        if self.dtype == "BF16":
            words = np.frombuffer(data, dtype="<u2").astype("<u4")
            floats = (words << 16).view("<f4")
        elif self.dtype == "F16":
            floats = np.frombuffer(data, dtype="<f2").astype(np.float32)
        else:
            floats = np.frombuffer(data, dtype="<f4").copy()
        return floats.reshape((count, self.hidden))

    def row(self, token_id: int):
        return self.rows(token_id, 1)[0]


def run_streamed_blocks(reader, layer_count: int, compute, hidden, *, pool_limit_bytes: int):
    """Synchronous single-block residency contract, usable in CPU-only tests."""
    if type(layer_count) is not int or not 1 <= layer_count <= 1000:
        raise ValueError("Invalid decoder-layer count")
    if type(pool_limit_bytes) is not int or pool_limit_bytes < 1:
        raise ValueError("Invalid packed-weight pool limit")
    for index in range(layer_count):
        with reader.cuda_block(index, pool_limit_bytes=pool_limit_bytes) as (operators, _pool):
            hidden = compute(index, operators, hidden)
    return hidden


def inspect_readiness(*, checkpoint=None, tokenizer=None, config=None, context=4096):
    """No GPU context, network, or weights are required for missing-file status."""
    if checkpoint is None:
        return {"status": "MISSING_CHECKPOINT", "can_chat": False,
                "trained_weights_verified": False,
                "message": "No trained Gemma 3 LittleBit-2 checkpoint supplied. Synthetic test weights are not a chatbot."}
    path = Path(checkpoint)
    if not path.exists():
        raise ChatUnavailable("Checkpoint path does not exist: " + str(path))
    cfg = read_gemma_config(config or test_fixture_path())
    try:
        report = build_report(cfg, context_tokens=context, checkpoint=path)
    except (ValueError, OSError) as exc:
        raise ChatUnavailable("Checkpoint format or memory preflight failed: " + str(exc)) from exc
    tok_path = Path(tokenizer) if tokenizer is not None else (
        path / "tokenizer.json" if path.is_dir() else path.parent / "tokenizer.json")
    present = tok_path.is_file()
    return {
        "status": ("PACKED_CHECKPOINT_UNVERIFIED" if present else "MISSING_TOKENIZER"),
        "can_chat": False, "can_attempt_unverified": bool(present),
        "trained_weights_verified": False,
        "tokenizer_file": str(tok_path),
        "checkpoint_preflight": report,
        "message": ("Structural metadata can pass without proving model quality. "
                    "To attempt inference on your own local weights, explicitly pass "
                    "--allow-unverified-checkpoint. Results may be nonsensical."),
    }


def _read_model_settings(config_path, arch: Gemma3TextArchitecture):
    path = Path(config_path)
    if path.stat().st_size > 128 * 1024:
        raise ChatUnavailable("Model config is oversized")
    try:
        data = json.loads(path.read_text(encoding="utf8"))["text_config"]
        rotary = data.get("rope_scaling") or {}
        settings = SimpleNamespace(
            hidden=arch.hidden, intermediate=arch.intermediate,
            q_heads=arch.q_heads, kv_heads=arch.kv_heads,
            head_dim=arch.head_dim, sliding_window=arch.sliding_window,
            eps=float(data["rms_norm_eps"]),
            query_pre_attn_scalar=float(data["query_pre_attn_scalar"]),
            local_rope_theta=float(data.get("rope_local_base_freq", 10000.0)),
            global_rope_theta=float(data["rope_theta"]),
            global_rope_factor=float(rotary.get("factor", 1.0)),
            attn_softcap=data.get("attn_logit_softcapping"),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ChatUnavailable("Config is missing required Gemma 3 numeric settings") from exc
    for attr in ("eps", "query_pre_attn_scalar", "local_rope_theta",
                 "global_rope_theta", "global_rope_factor"):
        v = getattr(settings, attr)
        if not math.isfinite(v) or v <= 0:
            raise ChatUnavailable("Invalid Gemma 3 config value " + attr)
    if settings.attn_softcap is not None:
        settings.attn_softcap = float(settings.attn_softcap)
        if not math.isfinite(settings.attn_softcap) or settings.attn_softcap <= 0:
            raise ChatUnavailable("Invalid attention softcap")
    return settings


class FullGemmaChatRuntime:
    """Experimental 62-block CUDA scheduler for an authentic local checkpoint.

    No real 27B checkpoint was available for end-to-end testing. No numerical
    parity/throughput claims can be made for genuine 27B weights yet.
    """

    def __init__(self, checkpoint, *, config=None, context=4096,
                 limits=None, allow_unverified=False):
        if not allow_unverified:
            raise ChatUnavailable("Checkpoint training quality is not verified. Explicitly pass --allow-unverified-checkpoint to test it.")
        self.config_path = Path(config or test_fixture_path())
        self.arch = read_gemma_config(self.config_path)
        if not self.arch.matches_target_27b:
            raise ChatUnavailable("Full-size runtime is restricted to the verified Gemma 3 27B text architecture")
        if type(context) is not int or not 1 <= context <= 4096:
            raise ChatUnavailable("Current GPU KV implementation requires context in 1..4096")
        self.context = context
        self.limits = (limits or MemoryLimits()).validate()
        try:
            self.preflight = build_report(self.arch, context_tokens=context,
                                          checkpoint=Path(checkpoint), limits=self.limits)
            self.reader = LargeGemmaCheckpoint(checkpoint, self.arch,
                      max_block_bytes=self.limits.max_host_block_mib * MIB)
            self.embeddings = CheckpointEmbeddings(self.reader.group,
                expected_vocab=self.arch.vocab, expected_hidden=self.arch.hidden)
            self.settings = _read_model_settings(self.config_path, self.arch)
            self.final_norm_cpu = self.reader.group.read(FINAL_NORM).astype(np.float32)
            if not np.isfinite(self.final_norm_cpu).all():
                raise ChatUnavailable("Final normalization tensor contains non-finite values")
        except (CheckpointFormatError, CheckpointRejected, OSError, ValueError) as exc:
            raise ChatUnavailable("Unable to load local packed checkpoint: " + str(exc)) from exc
        from cuda_packed import _cupy
        try:
            self.cp = _cupy()
            self.state_pool = self.cp.cuda.MemoryPool()
            self.state_pool.set_limit(size=self.limits.state_pool_mib * MIB)
            with self.cp.cuda.using_allocator(self.state_pool.malloc):
                self.final_norm = self.cp.asarray(self.final_norm_cpu, dtype=self.cp.float32)
        except Exception as exc:
            raise ChatUnavailable("CuPy CUDA initialization/state allocation failed: " + str(exc)) from exc
        self.weight_pool_bytes = self.limits.weight_pool_mib * MIB
        self.position = 0
        self.caches = []
        self.reset_session()

    def reset_session(self):
        from gemma3_cuda_decoder import CUDAGemma3KVCache
        kinds = self.arch.attention_pattern
        self.caches = [CUDAGemma3KVCache(kind, self.arch.sliding_window,
                                        max_tokens=self.context) for kind in kinds]
        self.position = 0
        # Release stale KV arrays before rebuilding a conversation from scratch.
        self.cp.cuda.get_current_stream().synchronize()
        self.state_pool.free_all_blocks()

    def _norms(self, index: int):
        prefix = f"language_model.model.layers.{index}."
        raw = {}
        for key in NORM_SUFFIXES:
            source = self.reader.group.read(prefix + key + ".weight")
            raw[key] = self.cp.asarray(source, dtype=self.cp.float32)
        return raw

    def _block(self, index, ops, hidden):
        from gemma3_cuda_decoder import _rmsnorm_cuda, _rope_cuda, _attention_cuda
        cp, cfg, pos = self.cp, self.settings, self.position
        with cp.cuda.using_allocator(self.state_pool.malloc):
            norms = self._norms(index)
            x = cp.asarray(hidden, dtype=cp.float32)
            normed = _rmsnorm_cuda(cp, x, norms["input_layernorm"], cfg.eps)
            q = ops["self_attn.q_proj"].forward(normed).reshape(cfg.q_heads, cfg.head_dim)
            k = ops["self_attn.k_proj"].forward(normed).reshape(cfg.kv_heads, cfg.head_dim)
            v = ops["self_attn.v_proj"].forward(normed).reshape(cfg.kv_heads, cfg.head_dim)
            q = _rmsnorm_cuda(cp, q, norms["self_attn.q_norm"], cfg.eps)
            k = _rmsnorm_cuda(cp, k, norms["self_attn.k_norm"], cfg.eps)
            is_local = self.arch.attention_pattern[index] == "sliding_attention"
            theta = cfg.local_rope_theta if is_local else cfg.global_rope_theta
            factor = 1.0 if is_local else cfg.global_rope_factor
            q = _rope_cuda(cp, q, pos, theta, factor)
            k = _rope_cuda(cp, k, pos, theta, factor)
            attended = _attention_cuda(cp, q, k, v, self.caches[index], pos, cfg).reshape(-1)
            projected = ops["self_attn.o_proj"].forward(attended)
            residual = x + _rmsnorm_cuda(cp, projected, norms["post_attention_layernorm"], cfg.eps)
            feed = _rmsnorm_cuda(cp, residual, norms["pre_feedforward_layernorm"], cfg.eps)
            gate = ops["mlp.gate_proj"].forward(feed)
            up = ops["mlp.up_proj"].forward(feed)
            activated = 0.5 * gate * (1.0 + cp.tanh(
                cp.float32(math.sqrt(2.0 / math.pi)) * (gate + cp.float32(.044715) * gate**3)))
            mlp = ops["mlp.down_proj"].forward(activated * up)
            return (residual + _rmsnorm_cuda(cp, mlp, norms["post_feedforward_layernorm"], cfg.eps)).astype(cp.float32)

    def forward_token(self, token_id: int):
        if type(token_id) is not int or not 0 <= token_id < self.arch.vocab:
            raise ValueError("Token ID outside Gemma vocabulary")
        if self.position >= self.context:
            raise ChatUnavailable("GPU context capacity reached")
        cp = self.cp
        try:
            with cp.cuda.using_allocator(self.state_pool.malloc):
                hidden = cp.asarray(self.embeddings.row(token_id), dtype=cp.float32)
                hidden = hidden * cp.float32(math.sqrt(self.arch.hidden))
            hidden = run_streamed_blocks(self.reader, self.arch.layers, self._block,
                                         hidden, pool_limit_bytes=self.weight_pool_bytes)
            self.position += 1
            return hidden
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise ChatUnavailable("GPU state memory budget exhausted") from exc

    def greedy_next(self, final_hidden):
        """Chunked tied output projection: no 2.6GB embedding GPU allocation."""
        from gemma3_cuda_decoder import _rmsnorm_cuda
        cp = self.cp
        with cp.cuda.using_allocator(self.state_pool.malloc):
            normalized = _rmsnorm_cuda(cp, final_hidden, self.final_norm, self.settings.eps)
        best_value, best_token = -math.inf, None
        # 1024 rows ≈ 22MB BF16 on disk/44MB float32, below one-block budget.
        for start in range(0, self.arch.vocab, 1024):
            count = min(1024, self.arch.vocab - start)
            with cp.cuda.using_allocator(self.state_pool.malloc):
                table = cp.asarray(self.embeddings.rows(start, count), dtype=cp.float32)
                scores = table @ normalized
                j = int(cp.argmax(scores).item())
                value = float(scores[j].item())
            if not math.isfinite(value):
                raise ChatUnavailable("Non-finite output logits from local checkpoint")
            if value > best_value:
                best_value, best_token = value, start + j
            del table, scores
        return best_token

    def generate(self, prompt_ids, *, max_new_tokens=16, stop_ids=()):
        if (not isinstance(prompt_ids, (tuple, list)) or not prompt_ids
                or any(type(x) is not int or not 0 <= x < self.arch.vocab for x in prompt_ids)):
            raise ChatUnavailable("Prompt has invalid Gemma token IDs")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
            raise ChatUnavailable("max_new_tokens must be 1..128")
        if len(prompt_ids) + max_new_tokens > self.context:
            raise ChatUnavailable("Prompt/generation would exceed configured context")
        self.reset_session()
        start = time.perf_counter()
        final_hidden = None
        for token in prompt_ids:
            final_hidden = self.forward_token(token)
        new = []
        for _ in range(max_new_tokens):
            candidate = self.greedy_next(final_hidden)
            if candidate in stop_ids:
                break
            new.append(candidate)
            if len(new) < max_new_tokens:
                final_hidden = self.forward_token(candidate)
        return {
            "generated_token_ids": new,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(new),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "checkpoint_quality_verified": False,
            "full_27b_gpu_execution_verified": False,
            "total_vram_peak_measured": False,
        }
