# Token-disjoint Gemma 3 layer-0 activation validation

## Why the previous 400-row result is not a generalization estimate

Gemma 3 layer-0 attention input is a deterministic function of a token ID:
original embedding row, embedding scale, then the layer-0 input RMSNorm.
Different prompts containing the same token therefore yield the same
pre-Q projection input. The earlier 140/130/130 prompt-based split held
out *prompts*, not tokens. It had only 118 unique embeddings across all
splits and permitted repeated activations (including BOS/control prefixes).

This milestone measures reconstruction on **token IDs not used for fitting**.

## Mechanism and scope

`capture-layer0 --token-disjoint`:

1. Tokenizes all user-provided prompt groups using the **matching original**
   Gemma tokenizer; does not load the 27B model or make external requests.
2. Forms the set of token IDs seen in each group. Drops every ID that occurs
   in two or more groups, including shared BOS and instruction delimiters.
3. Deduplicates IDs within each group, then selects up to `--max-rows` IDs
   per group using a stable SHA-256 seed-based ordering. Prompt order cannot
   change the split.
4. Refuses to write if any group has fewer than
   `--min-unique-tokens` exclusive IDs (default **64 per split**).
5. Reconstructs exactly one original layer-0 normalized embedding per
   selected token, reading only the required rows of the original weight
   shard. Stores both 2D activation matrices and aligned, signed int64
   `token_ids_calibration`, `token_ids_selection`, `token_ids_audit` in a
   versioned Safetensors bundle.

`activation-audit --require-token-disjoint` checks each token-ID array's
dtype, length, uniqueness, cross-group non-overlap, coverage metadata, and
cross-group activation-byte non-overlap. More importantly, it **replays
each token ID against the original source embedding and RMSNorm weights**,
requiring exact float32 equality with the supplied activation row. This
catches relabeled IDs or swapped rows rather than trusting file claims.
Auditing is read-only and performs NO SVD or training.

`activation-calibrate --require-token-disjoint` enforces the *same source
replay* **before** expensive factor initialization. It preserves the
existing held-out selection gate, untouched audit, strict same-BPW
contract and exact factor rollback on rejection. The final JSON discloses
actual source-replay verification and per-split token counts.

The old Safetensors capture remains usable for legacy, non-strict
calibration. It **cannot** be upgraded or marked token-disjoint by
editing metadata: its per-row token IDs were never stored.

## Step 1: Obtain a sufficiently diverse, independent text corpus

Use UTF-8 `calibration.txt`, `selection.txt`, `audit.txt`; each needs
3..512 unique lines. For useful confidence, supply many genuinely varied
prompts and vocabulary in each group: programming, math, science,
explanations, natural dialogue, uncommon words, numbers, and longer
passages. Do not duplicate prompts or fill files with repeated templates.
Keep the audit corpus reserved. A high number of lines is **not**
evidence of many unique token IDs; the command reports the true counts
AFTER dropping shared IDs.

**The previous eight prompts per group are unlikely to satisfy the
default 64-exclusive-token-per-split gate.** This is by design. To smoke
test the workflow, `--min-unique-tokens 16` is supported, but such a small
holdout is not suitable evidence of general model behavior.

## Step 2: Capture new, strictly token-disjoint data

```powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"

& $py littlebitchat.py train capture-layer0 `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 --projection self_attn.q_proj `
  --calibration-texts calibration.txt `
  --selection-texts selection.txt `
  --audit-texts audit.txt `
  --token-disjoint --min-unique-tokens 64 `
  --max-rows 512 --max-tokens 128 `
  --out training_runs\capture_layer0_q_token_disjoint.safetensors
```

If the command reports insufficient unique tokens, add more diverse
source texts, or verify the workflow with `--min-unique-tokens 16`.
Do not represent a smoke test as a research-quality validation.
Do not overwrite the earlier `capture_layer0_q.safetensors`.

## Step 3: Verify the new capture before any SVD

```powershell
& $py littlebitchat.py train activation-audit `
  --checkpoint models\gemma-3-27b-it `
  --activation-data training_runs\capture_layer0_q_token_disjoint.safetensors `
  --layer 0 --projection self_attn.q_proj `
  --require-token-disjoint
```

Require `token_disjoint_verified: true`,
`source_reconstruction_verified: true`, and nonzero disjoint
`unique_token_counts` for all three sets.

## Step 4: Strict 0.55-BPW calibration

```powershell
& $py littlebitchat.py train activation-calibrate `
  --checkpoint models\gemma-3-27b-it `
  --activation-data training_runs\capture_layer0_q_token_disjoint.safetensors `
  --require-token-disjoint `
  --layer 0 --projection self_attn.q_proj `
  --rank 622 --max-bpw 0.55 `
  --itq-iterations 20 `
  --svd-oversample 32 --svd-power-iters 5 `
  --refine-output-scales `
  --binary-refine-channels 64 --binary-refine-passes 3 `
  --min-relative-improvement 0.001 `
  --out training_runs\layer0_q_token_disjoint_calibration
```

**Interpretation:** Compare `selection_nrmse_before` vs `after`
and separately `audit_nrmse_before` vs `after` **within this
new dataset**. Because the token distribution changed, do not directly
compare the new audit percentage to the previous prompt-overlap audit.
An accepted selection candidate can still regress on untouched audit;
that regression remains visible, not hidden or used to tune the model.
Do not rerun repeated hyperparameter sweeps on the same audit set.

## Boundaries and evidence

- This protects against repeated token IDs and exact activation rows
  across the three source groups; the source checkpoint and tokenizer
  still have *unverified external provenance*.
- Source replay verifies exact local embedding/RMSNorm math, but not
  numerical parity with a complete original-model forward pass.
- Token IDs can be correlated semantically; token-disjoint is stricter
  than prompt-disjoint, not a guarantee of representative distribution.
- Layer 0 q/k/v only. No layer-1+, MLP, or end-to-end language benchmark.
- Original factors, residual path, physical bit budget, fallback,
  selection policy, and packed export format are unchanged.
- This capture may contain information about the source prompts.
  Handle both the data and prompt files as private.

Review the regression tests in `training/tests/test_token_disjoint.py`.
