# Activation-aware LittleBit calibration (opt-in, same BPW)

This pipeline calibrates the **existing signed-factor output-row scales** against
real input activations entering a chosen original Gemma linear projection. It
does not change factor signs, ranks, tensor names, or the physical BPW estimate.

The objective is not random-probe weight reconstruction: for captured inputs
`X`, each output row fits

```text
min_(g_primary >= 0, g_residual >= 0)
    || X W_teacher^T - g_primary (X W_primary^T)
                     - g_residual (X W_residual^T) ||_2^2
```

It uses a disjoint **three-way split**: calibration fits the candidate,
selection chooses deployment/fallback, and audit is untouched until AFTER
selection. A candidate that fails the selection improvement threshold is
discarded, restoring the exact original factors. The independent audit can
still show a regression, which is reported rather than hidden or retroactively
used for selection. No language benchmark is run.

## 1. Capture source activations from the ORIGINAL model

**Prerequisite:** A genuine complete original Gemma 3 27B checkpoint and enough
RAM to load it as BF16. Roughly 27 billion parameters at 2 bytes each means
~54 GB of raw weights **before** runtime overhead. A 6 GB laptop GPU does not
satisfy that requirement by itself. Run capture on a high-RAM CPU workstation
or suitable compute instance if your laptop cannot load the original model.
Capture is never automatic. The 27B loader has not been validated on your
specific hardware.

On the capture machine, optionally install `transformers` and `accelerate` in
the selected virtual environment. The tool never downloads model weights;
`local_files_only=True` is enforced.

Prepare three UTF-8 text files with one prompt per line, at least three
distinct prompts each, and no prompt repeated between files:
`calibration.txt`, `selection.txt`, `audit.txt`. Keep their content representative
of the tasks your model will perform, and keep the audit prompts isolated from
tuning. Prompt data stays local.

```powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py littlebitchat.py train capture-activations `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 --projection self_attn.q_proj `
  --calibration-texts calibration.txt `
  --selection-texts selection.txt `
  --audit-texts audit.txt `
  --max-rows 512 --max-tokens 128 `
  --out training_runs\capture_layer0_q.safetensors `
  --confirm-full-model-memory
```

The capture uses a PyTorch module `forward_pre_hook` on the **actual linear
projection**. It verifies the local full model's weight matches the separately
read teacher matrix byte-for-byte after FP32 conversion. It stops each forward
once the hook has captured the input, avoiding expensive later decoder
computation. Three separate splits are saved as `calibration`, `selection`,
and `audit` tensors inside Safetensors, with binding metadata and source
prompt hashes. Text data is not saved.

The dataset file may contain private information encoded in hidden states.
Treat it as sensitive and do not publish it.

**Provenance limitation:** The metadata provides teacher checksum and
source-declared capture provenance; neither the capture file nor these
checksums independently prove which software, model, or corpus produced the
activations. All reports therefore mark external provenance as unverified.

For custom pretrained-model runners, the tested Python helper is
`training.activation_capture.capture_projection_inputs(model, teacher, ...)`.
The supplied model must be loaded in `.eval()` mode and expose the expected
module path.

## 2. Calibrate and compare at exactly the same BPW

Only after a real capture has been obtained, run:

```powershell
& $py littlebitchat.py train activation-calibrate `
  --checkpoint models\gemma-3-27b-it `
  --activation-data training_runs\capture_layer0_q.safetensors `
  --layer 0 --projection self_attn.q_proj `
  --rank 622 --max-bpw 0.55 `
  --itq-iterations 20 --svd-oversample 32 --svd-power-iters 5 `
  --refine-output-scales `
  --binary-refine-channels 64 --binary-refine-passes 3 `
  --min-relative-improvement 0.001 `
  --out training_runs\layer0_q_activation
```

The existing best initialization is first constructed using the same SVD,
scale, and binary sign options as before. **Only the existing `u1` and
`u1_R` output-row magnitude vectors** are calibrated; no extra sign bits
are added. The reported estimated effective BPW must not exceed `--max-bpw`.
If selection NRMSE fails the required relative improvement threshold,
`FALLBACK_LEGACY` is exported (the untouched original factors).

Reports include calibration, selection and untouched audit NRMSE before/after,
the activation bundle SHA-256, exact teacher SHA-256, the physical BPW
budget check, packed inference parity, and the fallback decision.
Only one output file is created per destination; existing exports are never
overwritten. Training and full-model language quality are explicitly **not**
claimed. Check the independent audit result even after a selection win.

**No activation data?** The command fails closed instead of silently using
random Gaussian probes. The older `train diagnose` remains available
unchanged for weight-space evaluation.

## Verification and limitations

- Synthetic unit tests check teacher binding, separate prompt splits, input
  width, nonfinite rejection, actual forward hook inputs, recovered row
  scales, fallback restoration, same BPW, packed export, and CLI failure mode
- A positive result on a single linear projection does not establish
  end-to-end Gemma 3 27B inference, calibration across other layers, or
  language quality
- Large original checkpoints may not fit the source machine's RAM

Primary references: PyTorch `register_forward_pre_hook` documentation and
Hugging Face Safetensors format. This is a project-specific optimizer,
**not** an implementation of the published AWQ/GPTQ algorithms.
