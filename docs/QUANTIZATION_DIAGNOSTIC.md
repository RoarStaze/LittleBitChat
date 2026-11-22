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
