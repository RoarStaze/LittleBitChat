"""Deterministic token-identity holdout partition for Gemma layer-0 inputs.

At layer 0, identical token IDs produce identical embedding/RMSNorm inputs.
A prompt-disjoint corpus is therefore NOT an independent activation split.
Exclude all token IDs shared across prompt groups and sample exactly one
activation row per remaining token ID, deterministically and without leakage.
"""
from __future__ import annotations

import hashlib


SPLITS = ("calibration", "selection", "audit")


class TokenDisjointError(ValueError):
    """The supplied text does not support a credible token-disjoint holdout."""


def select_token_disjoint_ids(tokenizer, prompts, *, max_tokens, max_rows,
                              min_unique_tokens=64, vocab_size, seed=42):
    """Plan a strict 3-way split BEFORE reading any teacher embeddings.

    Each group retains only token IDs never encountered in the other
    groups. Repeated control/prefix tokens are intentionally excluded.
    Selection is stable over prompt order and uses a fixed cryptographic
    hash, avoiding first-512-token and prompt-order bias.
    """
    for name, value, minimum, maximum in (
            ("max_tokens", max_tokens, 16, 512),
            ("max_rows", max_rows, 16, 8192),
            ("min_unique_tokens", min_unique_tokens, 16, 8192),
            ("vocab_size", vocab_size, 1, 10000000)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise TokenDisjointError(f"{name} must be {minimum}..{maximum}")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise TokenDisjointError("seed must be a nonnegative integer below 2**31")
    if max_rows < min_unique_tokens:
        raise TokenDisjointError(
            f"max_rows={max_rows} is below min_unique_tokens={min_unique_tokens}")
    if not hasattr(tokenizer, "encode_conversation"):
        raise TokenDisjointError("Matching Gemma tokenizer required")
    if not isinstance(prompts, dict) or set(prompts) != set(SPLITS):
        raise TokenDisjointError("Expected exactly calibration, selection and audit text groups")
    encountered={}
    for name in SPLITS:
        values=prompts[name]
        if (not isinstance(values,list) or not 3<=len(values)<=512
                or any(not isinstance(p,str) or not p.strip() for p in values)
                or len(set(values)) != len(values)):
            raise TokenDisjointError(f"Invalid/duplicate {name} prompts")
        tokens=set()
        for prompt in values:
            sequence=tokenizer.encode_conversation([("user",prompt)])
            ids=list(sequence)[:max_tokens]
            if (not ids or any(type(t) is not int or not 0<=t<vocab_size
                               for t in ids)):
                raise TokenDisjointError(
                    "Invalid tokenizer ID or source embedding vocabulary mismatch")
            tokens.update(ids)
        encountered[name]=tokens
    if len(set(sum((prompts[name] for name in SPLITS),[]))) != sum(
            len(prompts[name]) for name in SPLITS):
        raise TokenDisjointError("Prompt overlap across splits")

    shared=(encountered["calibration"] & encountered["selection"]
            | encountered["calibration"] & encountered["audit"]
            | encountered["selection"] & encountered["audit"])
    exclusive={name:(encountered[name]-shared) for name in SPLITS}
    eligible={name:len(exclusive[name]) for name in SPLITS}
    if any(count < min_unique_tokens for count in eligible.values()):
        raise TokenDisjointError(
            "Insufficient unique token-disjoint IDs after removing shared "
            f"tokens: {eligible}; require at least {min_unique_tokens} in "
            "EVERY split. Supply more diverse, non-overlapping prompts.")

    def ordering(token_id):
        digest=hashlib.sha256(
            f"littlebit-token-disjoint-v1:{seed}:{token_id}".encode("ascii")).digest()
        return digest,token_id

    chosen={name:sorted(exclusive[name],key=ordering)[:max_rows] for name in SPLITS}
    assert all(not set(chosen[a]).intersection(chosen[b])
               for a,b in (("calibration","selection"),
                           ("calibration","audit"),("selection","audit")))
    return {
        "ids":chosen,
        "shared_dropped":len(shared),
        "source_unique_counts":{name:len(encountered[name]) for name in SPLITS},
        "eligible_unique_counts":eligible,
        "sampled_unique_counts":{name:len(chosen[name]) for name in SPLITS},
        "selection_algorithm":"sha256_token_id_seeded_v1",
        "seed":seed,
        "min_unique_tokens":min_unique_tokens,
    }
