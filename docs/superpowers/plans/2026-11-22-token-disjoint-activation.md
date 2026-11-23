# 2026-11-22 — Token-disjoint activation validation implementation plan

## Goal

Replace the previous pseudo-held-out layer-0 activation evaluation (distinct
prompts but repeated token embeddings) with a truly token-ID-disjoint, auditable
evaluation at unchanged estimated 0.55 BPW.

## Engineering decisions

- Retain the original layer-0 capture command and its data schema by default.
- Add `--token-disjoint` to produce a **versioned** Safetensors capture with
  aligned `token_ids_<split>` arrays. New files, never overwrite old ones.
- For each prompt split, tokenize all inputs with the original local tokenizer,
  remove any ID appearing in another split, deduplicate, and sample IDs by
  deterministic SHA-256 ranking independent of prompt order.
- Require at least 64 exclusive tokens per split by default; fail closed with
  actual eligible counts rather than using repeated rows as extra samples.
- Preserve the previous 3-way calibration / selection / untouched audit split.
- Add `activation-audit` and `--require-token-disjoint` gating, bound to the
  original local teacher SHA-256.
- Verify *both* token-label non-overlap and exact replay of every activation
  from the original local embedding and RMSNorm files. No full-model loading.
- Keep original SVD/ITQ initializer, row-scale optimizer, 0.55-BPW policy,
  packed inference parity, and exact fallback on failed selection untouched.
- Clearly disclose that a source-weight replay is **not** independently
  authenticated original-model provenance or a language benchmark.

## TDD verification

1. Partition algorithm: deterministic, duplicate-free, no shared token IDs,
   too-small datasets fail.
2. Capture: one unique token per row, expected exact original inputs, original
   teacher hash and versioned metadata; legacy format still works.
3. Tampering: reject duplicated IDs across splits, missing/misaligned IDs,
   replaced labels with original rows, and invalid source.
4. Audit: read-only CLI checks before SVD; strict old captures rejected.
5. Calibration: strict v2 accepted or falls back at same BPW, tests assert
   source replay, unchanged packed export correctness.
6. Run all GitHub CI jobs against exact final PR head and only then merge.
7. User workflow: `git pull --ff-only origin main`, new token-disjoint capture,
   read-only audit, strict calibration to a fresh directory.

## Limitations

Token-disjoint held-out lexical embeddings cannot prove full-model reasoning
or generation quality. The user's earlier eight prompts per split may not
meet the recommended default 64 unique IDs each; request broader text rather
than lowering the threshold without disclosure.
