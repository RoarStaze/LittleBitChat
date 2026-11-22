# Gemma 3 LittleBit-2: resumable multi-projection training

This milestone provides an **automatic sequential trainer for locally supplied pretrained dense Gemma 3 27B projection weights**, NOT full-model training or an automatically downloaded model.

## Verified goal

- Inspect original Gemma 3 Safetensors tensor **headers** across selected decoder layers, without loading entire dense matrices.
- Verify expected 27B tensor dimensions using the local reviewed Gemma 3 reference config. Test-only non-27B shapes require an explicit test switch.
- Train each selected projection using the existing SVD → Joint-ITQ → SmoothSign QAT / optional residual algorithm.
- Load and train **one projection at a time**. Do not hold multiple dense projections in memory.
- Export actual learned Samsung-compatible U/V packed factors, magnitude scales and optional residuals into per-projection Safetensors files.
- Validate packed export through the single-layer math and record original-teacher hash, holdout normalized RMSE, effective BPW, training settings and output SHA256.
- Record completed jobs in a JSON batch manifest written **atomically after each success**.
- Resume without rerunning already-verified completed jobs. Fail closed on changed training settings, a modified completed output, a changed teacher tensor, unsafe shard references or a missing original source.
- Reject a non-finite quality threshold. Report whether each projection passes the **random-probe RMSE** threshold, but **never claim validated language quality**.

**Still not accomplished:** Activation-aware distillation, model-wide QAT, assembly of a complete decoder/text-embedding/norm checkpoint, measured full-size inference or a trained 27B chat model.

## First action: scan your original local dense Gemma 3 checkpoint

No LittleBit trained checkpoint is required; however, the actual original pretrained Gemma 3 27B teacher must be present locally. The scanner does not download weights or allocate a 27B model.

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py -m pip install -r requirements-training.txt

& $py littlebitchat.py train inventory --checkpoint "D:\Models\gemma-3-27b-it" --layer-start 0 --layer-end 1
~~~

The real 27B teacher must contain the original HF tensor names. The inventory verifies the seven original dense projection dimensions for decoder layer zero and reports total dense bytes. It intentionally does **not** generate any packed trained model yet.

If the original teacher is not available on disk, the command explicitly fails with a missing checkpoint error. Do **not** point it to your previously generated synthetic LittleBit projection; that is neither the original dense pretrained 27B model nor an adequate teacher.

## Begin real layerwise QAT training

Default **one projection per invocation** to keep memory usage bounded and give you a chance to evaluate the first trained layer:

~~~powershell
& $py littlebitchat.py train batch --checkpoint "D:\Models\gemma-3-27b-it" --layer-start 0 --layer-end 1 --projections self_attn.q_proj --rank 32 --steps 50 --device cpu --max-jobs 1 --out training_runs\gemma-27b
~~~

Every layer emits an individual packed Safetensors tensor, a layer audit JSON, a resumable PyTorch state file and the batch_manifest.json progress file.

To select all seven projections from the first decoder layer, start with a **fresh empty output directory** (different selection settings are deliberately incompatible with resuming a previous batch):

~~~powershell
& $py littlebitchat.py train batch --checkpoint "D:\Models\gemma-3-27b-it" --layer-start 0 --layer-end 1 --rank 32 --steps 50 --device cpu --max-jobs 7 --out training_runs\gemma-layer0
~~~

To resume the **same selection/configuration** after interruption and skip completed checksummed exports, rerun with --resume:

~~~powershell
& $py littlebitchat.py train batch --checkpoint "D:\Models\gemma-3-27b-it" --layer-start 0 --layer-end 1 --rank 32 --steps 50 --device cpu --max-jobs 7 --out training_runs\gemma-layer0 --resume
~~~

For a larger 62-layer sweep, use --layer-start 0 --layer-end 62 with the same output directory throughout and use --max-jobs to control per-invocation work. **Changing selected layers/projections, hyperparameters or context and using the same saved manifest intentionally fails.** Use a different output directory for a new experiment.

If training is interrupted *within* a projection and the intermediate .state.pt file exists, --resume hands that state to the existing QAT trainer and continues from its optimizer checkpoint (assuming its checkpoint step is below the requested final step). If the process died after the final step state was saved but before the audit was committed, the batch may require recovery or an increased final step count; do not delete valid learned weights without first backing them up.

## Quality and compression metrics

The user’s first 50-step toy demo achieved holdout normalized RMSE ≈ 0.615 and effective BPW ≈ 3.83, not a good approximation or a 0.55 BPW result. For real weights, each exported projection reports:
- trained vs initial held-out random-input RMSE and normalized RMSE
- actual effective BPW at the chosen rank, not merely target BPW
- binary-factor inference parity with the exported checkpoint
- SHA-256 for the exact original teacher matrix and exported packed file
- the random-probe gate threshold (default 0.25) and per-layer pass/fail

A good random-probe error is **not** evidence of Gemma 3’s real language quality. For that, full-model calibration, distillation and downstream evaluation are mandatory.

## Hard constraints

The full original Gemma 3 27B checkpoint is roughly 54 GB of BF16 parameter weights and must be separately acquired under the original model license. Download and disk access are not automated. Your laptop has only 6 GB VRAM and about 16 GB RAM. This trainer uses one original projection at a time and can run in CPU mode, but **processing 434 such matrices and preserving Gemma 3’s language quality is not established or guaranteed feasible on this laptop**.

A complete 27B model also includes embeddings, norms, and model-wide calibration. Per-projection packed results **must not** be passed to LittleBitChat's chat loader as a complete model.

## Tests and safety

GitHub Actions runs the migrated engine/chat suites plus new batch tests. Test fixtures use small real-format BF16 tensors, a deliberate --allow-test-shapes override, actual PyTorch QAT and packed-output verification. The production CLI defaults to strict Gemma 3 27B shapes.

No KalshiEdge files are modified. No cloud services are invoked. Trained factors and optimizer state remain local under the ignored training_runs directory.
