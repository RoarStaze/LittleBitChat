# Diagnosing original Gemma 3 projection compression error

A real Gemma 3 27B layer-0 q_proj experiment with rank 128, 200 steps, learning rate 0.0001 and a genuine Google BF16 shard showed a small improvement: independent audit RMSE 0.575456 to 0.572126, NRMSE 0.756057, actual 0.12407 bits/weight. The numeric quality gate (0.25) still failed.

The next question is not another optimizer hyperparameter: **How much error is caused by low-rank approximation, and how much is introduced by binary factorization?**

## Diagnose without retraining or downloading more weights

PowerShell:

    cd "$HOME\Desktop\LittleBitChat"
    git pull --ff-only origin main
    $py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
    & $py littlebitchat.py train diagnose --checkpoint models\gemma-3-27b-it --layer 0 --projection self_attn.q_proj --rank 128 --itq-iterations 20 --samples 96

The command reads one original dense BF16 teacher projection into CPU memory. It evaluates two initialization paths:
- The randomized low-rank SVD approximation, *before any binarization*.
- The LittleBit-2 SVD/Joint-ITQ hard-sign and magnitude-factor approximation, with an optional residual path, exactly as initialized for the existing QAT training.

It reports relative teacher-matrix error and independent random-probe projection-output NRMSE for each, plus teacher hash, rank, estimated bits/weight, and explicit no-training/no-full-model/no-language-quality flags.

For the original trainer's seed=42, the diagnostic's default samples=96 and audit seed follow the same deterministic convention used to create the previous checkpoint's audit probes. The diagnostic itself does not modify training history.

## Interpret the evidence

- **Large SVD error** suggests the low-rank bottleneck; optimizer steps cannot fix limited representational capacity.
- **Small SVD error but large binary error** suggests investigating Joint-ITQ rotation, scale vectors and hard-sign learning.
- **Small errors on random Gaussian probes** still do NOT establish meaningful Gemma text quality, which requires real hidden-state calibration and model-wide evaluation.

Important: This is PyTorch's *randomized* SVD approximation, not the mathematically exact best-rank SVD. Its error is not a proven lower bound. The binary factor may also use a second residual path, so the two errors are NOT additive.

## Safety and next milestone

The tool does not download more shards, write an output file, retrain, change checkpoints, or make a chatbot. It performs potentially expensive CPU calculations on the 4096x5376 real teacher projection; start at rank 128 rather than the much larger target rank 623.

Use its measured breakdown to prioritize representation rank versus binary quantization improvements. To ultimately run a good 27B chat model we still need calibration on actual model activations, trained full-model factors and a complete evaluated checkpoint.


## Fixed-BPW binary output-row refinement (opt-in)

The new `--refine-output-scales` mode fits the existing `u1` and optional
`u1_R` row-scale vectors **jointly**, at fixed sign patterns, factor rank,
and packed storage cost. For each output row it solves nonnegative 2-variable
least squares against the original dense projection, considering the interior,
single-path boundaries, zero, and unchanged baseline. It restores the
original scales if FP32 reconstruction numerically regresses.

Compare the previous initialization with the refined version, without
training or changing the checkpoint:

```powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py littlebitchat.py train diagnose `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 `
  --projection self_attn.q_proj `
  --rank 622 `
  --itq-iterations 20 `
  --samples 96 `
  --refine-output-scales
```

The `binary_initial_*` metrics remain the unchanged baseline, and
`binary_refined_*` metrics report the recalibrated factors. Diagnostic
Gaussian probes are **not** used to select scales; the dense teacher is.
A lower dense-weight RMSE does not guarantee lower independent probe NRMSE
or better language quality. This option adds CPU/GPU work but no stored bits.

To use the improved initialization in a **new** QAT job (fresh output path):

```powershell
& $py littlebitchat.py train batch `
  --checkpoint models\gemma-3-27b-it `
  --layer-start 0 --layer-end 1 `
  --projections self_attn.q_proj `
  --rank 128 --steps 200 --device cpu --max-jobs 1 `
  --refine-output-scales --out training_runs\gemma-row-refined
```

The flag also works with `train layer` and `train demo`. It is deliberately
opt-in to preserve established runs. Training state records the policy and
refuses a resume if the flag changes; batch manifests also record it when
enabled. Do not claim a trained full model or language improvement from
single-layer reconstruction metrics. The official teacher's provenance must
still be verified independently.


## Greedy binary sign-coordinate refinement (opt-in)

For a stronger representation search than row-scale-only fitting, use
`--binary-refine-channels 8`. This directly optimizes selected *binary*
signs in both U and V factor matrices. It chooses the highest-energy
components across both paths and alternates exact row-wise and column-wise
sign decisions for each selected channel, holding the others fixed.
It accepts a channel only if dense teacher squared error decreases. A final
reconstruction safeguard rolls back all sign updates on numerical regression.
Factor magnitudes, tensor layout, rank, and estimated BPW remain unchanged.

First evaluate (no training, no outputs written):

```powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py littlebitchat.py train diagnose `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 --projection self_attn.q_proj `
  --rank 622 --itq-iterations 20 --samples 96 `
  --refine-output-scales --binary-refine-channels 8
```

Compare `binary_initial_*`, `binary_refined_*` (row scales),
and `binary_sign_refined_*` (new signed-factor optimization). The
`binary_refine_details` object reports accepted versus examined channels
and dense reconstruction SSE. The unchanged independent Gaussian probes are
**evaluation only**, not optimization input; an improvement in dense SSE
does not guarantee improved probe NRMSE or language capabilities.
Do not claim the method beats LittleBit-2 on language benchmarks without
activation calibration, full-model training and an actual held-out evaluation.

To opt in for a *new* QAT run, supply both flags to `train layer` or
`train batch`, using a fresh `--out` path. Run `train diagnose` first:
more channels cost more time and temporary dense teacher memory. The
safety limit is 0..256 channels per projection. Both flags are recorded
in training checkpoints and in batch manifests; any different resume
policy is rejected. Legacy runs retain the default 0-channel behavior.


## Stronger mathematical foundation: guarded SVD + multi-pass binary optimization

This opt-in experimental candidate addresses two weaknesses that were visible
in the single-projection rank-622 diagnosis: randomized SVD truncation error
and a one-pass local minimum in binary-sign refinement.

**Guarded randomized SVD.** The legacy initializer uses `q=rank` and
`niter=3` in `torch.svd_lowrank`. The advanced initializer generates
a second sketch with `q=min(rank+oversample, smaller_dimension)` and a
configurable number of power iterations, then keeps the candidate only
when its *rank-truncated dense teacher weight SSE* is lower. Because the
best real-valued approximation can produce worse binary signs, the
complete two-path binary initializer is also compared against the
unchanged legacy binary initialization; the lower-SSE result is used.
This compares teacher *weight* errors, not activation loss or language
quality. The method is randomized, not an exact SVD lower bound.

**Multi-pass binary coordinate descent.** With
`--binary-refine-passes N`, the same selected U/V sign channels in the
primary and residual paths are revisited up to N times, retaining only
lower-SSE coordinate updates and stopping if no changes are accepted.
The full initial sign state is preserved for rollback. Diagnostics expose
`passes_requested`, `passes_completed`, `accepted_updates`, and
`accepted_channels`. With the default `N=1`, earlier runs retain their
previous initialization.

For your real Gemma 3 27B layer-0 q_proj, compare the **unchanged legacy
baseline** against the enhanced stages with one read-only command:

```powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py littlebitchat.py train diagnose `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 --projection self_attn.q_proj `
  --rank 622 --itq-iterations 20 --samples 96 `
  --svd-oversample 32 --svd-power-iters 5 `
  --refine-output-scales `
  --binary-refine-channels 64 --binary-refine-passes 3
```

The `svd_relative_weight_rmse` and `binary_initial_relative_weight_rmse`
remain the original rank-622 baseline. Enhanced SVD values appear under
`svd_enhanced_*`, the guarded two-path binary starting point under
`binary_svd_enhanced_*`, row-scale fitting under `binary_refined_*`,
and multi-pass sign fitting under `binary_sign_refined_*`.

Important: this command can be substantially slower and use more temporary
CPU RAM than the earlier diagnostic. Enhanced initialization may compute
both legacy and enhanced SVDs for the main and residual factors and select
the lower-loss *whole* binary candidate; the diagnostic additionally
computes the independent legacy baseline for comparison. No training,
downloads, checkpoint changes, or model assembly are performed.

If testing subsequent QAT, provide the same flags to a **new**
`train layer` or `train batch` output directory, never reuse an older
checkpoint with different policy. The implementation rejects incompatible
resumes and preserves existing defaults and packed BPW. Do not conflate
weight-RMSE improvements with coherent Gemma 3 chat capabilities: real
hidden-state activation calibration and model-wide validation remain
outstanding.

References for randomized SVD (principles, not empirical claims about
this checkpoint): PyTorch `torch.svd_lowrank` documentation,
https://docs.pytorch.org/docs/stable/generated/torch.svd_lowrank.html ;
Halko, Martinsson & Tropp, *Finding Structure with Randomness*, SIAM
Review 53(2), 2011, doi:10.1137/090771806.
