# Large-scale token-disjoint validation with a frozen audit

This benchmark is for **Gemma 3 27B layer-0 q/k/v projection input reconstruction**.
It requires only the original indexed embedding, first input RMSNorm and
selected projection weights plus the original local `tokenizer.json`.
It **does not** load 27B weights, download anything or evaluate language
generation.

## Why another capture?

Previous prompt-derived capture contained **27/29/33** truly exclusive
token IDs across fit, selection and audit. It showed an encouraging ~22%
relative audit NRMSE reduction but was too small to provide a stable
population estimate. Here the default capture measures **2,048 fit**,
**1,024 selection**, and **1,024 frozen audit** different token IDs.

With the default **uniform tokenizer vocabulary type sampling**, every
eligible distinct tokenizer ID receives equal sampling weight, regardless
of its natural language frequency. All known `<...>` control, reserved
and image-token IDs are excluded. This is **broad coverage of token
types**, not a representative text-use distribution. For a more natural
candidate pool, explicitly provide `--source-text-file path-to-local-corpus.txt`;
that samples unique token types encountered in the local text only
(still not token-frequency-weighted). Do not treat either as a downstream
language-quality test.

## Freeze once, then reuse the exact audit

```powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"

& $py littlebitchat.py train capture-vocabulary `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 --projection self_attn.q_proj `
  --calibration-tokens 2048 --selection-tokens 1024 --audit-tokens 1024 `
  --sample-seed 42 --audit-seed 20261122 `
  --frozen-audit-lock training_runs\frozen_layer0_q_audit.lock.json `
  --out training_runs\capture_layer0_q_vocab_v1.safetensors
```

The first successful capture creates a **new** lock with exclusive-file
creation and never overwrites it. The lock binds original teacher SHA-256,
full original tokenizer vocabulary digest, the set of audit IDs, and
the exact source-reconstructed audit activation bytes. Its own SHA-256
is included in the capture metadata and result JSON. Preserve a copy of
the lock and its reported SHA-256 in a trusted place outside experimental
output directories. In subsequent commands, use
`--expected-audit-lock-sha256 <the-original-64-character-hash>` to
**pin** that fingerprint independently. This is optional but strongly
recommended: without an external trusted SHA, the same untrusted
filesystem cannot defend against an attacker editing both lock and capture.

Change only `--sample-seed` and output filename to make a new
calibration/selection sample that preserves the exact frozen audit:

```powershell
& $py littlebitchat.py train capture-vocabulary `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 --projection self_attn.q_proj `
  --calibration-tokens 2048 --selection-tokens 1024 --audit-tokens 1024 `
  --sample-seed 99 --audit-seed 20261122 `
  --frozen-audit-lock training_runs\frozen_layer0_q_audit.lock.json `
  --out training_runs\capture_layer0_q_vocab_v2.safetensors
```

The token-ID sets for fitting/selection can change, but the audit
token IDs and reconstructed activations **must stay byte-identical**.
The command rejects changed lock parameters, original projection
binding, changed audit values, missing source weights, overlapping
tokens or any attempted output/lock overwrite. Keep the original
`capture_layer0_q_token_disjoint.safetensors` separate.

## Verify BEFORE training

```powershell
& $py littlebitchat.py train activation-audit `
  --checkpoint models\gemma-3-27b-it `
  --activation-data training_runs\capture_layer0_q_vocab_v1.safetensors `
  --layer 0 --projection self_attn.q_proj `
  --require-token-disjoint `
  --frozen-audit-lock training_runs\frozen_layer0_q_audit.lock.json
```

Require `token_disjoint_verified: true`,
`source_reconstruction_verified: true`, and
`frozen_audit_verified: true`.
`unique_token_counts` should be `2048/1024/1024` with defaults.
The source-weight replay uses original local embedding and RMSNorm
rows for *all* three splits. It checks token labels and exact
float32 matrix equality. This does not prove the true original 27B
forward pass was run.

## Run the same 0.55-BPW comparison

```powershell
& $py littlebitchat.py train activation-calibrate `
  --checkpoint models\gemma-3-27b-it `
  --activation-data training_runs\capture_layer0_q_vocab_v1.safetensors `
  --require-token-disjoint `
  --frozen-audit-lock training_runs\frozen_layer0_q_audit.lock.json `
  --layer 0 --projection self_attn.q_proj `
  --rank 622 --max-bpw 0.55 `
  --itq-iterations 20 `
  --svd-oversample 32 --svd-power-iters 5 `
  --refine-output-scales `
  --binary-refine-channels 64 --binary-refine-passes 3 `
  --min-relative-improvement 0.001 `
  --out training_runs\layer0_q_large_scale_v1
```

Use a new `--out` directory for each explicit experiment. This
retains **selection-only** candidate acceptance, same estimated
BPW (~0.54978), unchanged binary factor shapes, packed CPU
parity, and exact rollback if selection NRMSE fails to improve.

On a frozen audit, the report adds `audit_relative_nrmse_improvement_pct`,
`audit_relative_improvement_ci95_pct`, and a deterministic
**1,000-resample paired bootstrap** of error by unique held-out
token ID. This CI is **conditional on the sampled token-type
distribution**; it does not establish confidence about natural text,
semantic generalization, or full-model behavior.

## Audit discipline

Treat the lock as a **one-time reserved holdout**: decide rank,
refinement settings, and candidate-selection policy using calibration
and selection only; evaluate audit after locking those choices.
Repeatedly trying alternatives and preferring the one with the best
audit score invalidates its independent confirmatory interpretation.
Future experiments can reuse the frozen audit for descriptive
comparability, but not for repeated optimization or fresh unbiased
significance tests.

The earlier 27/29/33 result and this vocabulary-uniform result use
different input distributions; compare before/after **within each
dataset**, not raw cross-dataset NRMSE.

The tool does not independently authenticate the source checkpoint
or validate generated language quality. User-side successful full
27B testing remains a separate, resource-intensive milestone.
