# LittleBit-2 rank and QAT quality diagnostics

This milestone addresses an actual user-run result from a **real original Gemma 3 27B BF16 q_proj**:

| Metric | User-run result before quality guard |
|---|---:|
| Layer / module | 0 / self_attn.q_proj |
| Rank / residual factors | 32 / yes |
| Actual effective bits per original weight | 0.041341 |
| Initial random-input holdout RMSE | 0.630377 |
| Final random-input holdout RMSE | 0.664706 (about 5.45% worse) |
| Normalized RMSE | 0.892848 |
| Packed inference parity | Passed |
| Random-input quality gate | Failed (threshold 0.25) |

**Packed parity is not teacher fidelity.** The exported packed values faithfully reproduced the trained factors, but the factors did not represent the teacher well.

## Root cause and policy correction

The previous trainer exported the **final optimizer step regardless of held-out degradation**, and displayed only one random-input holdout score. A low rank is another plausible major source of approximation error. Neither fact alone establishes how much recovered language quality a real LLM will retain.

The new trainer:

1. Uses deterministic **validation probes** for periodic checkpoint selection. Candidate scores are evaluated at every --checkpoint-every steps and at the end, with initialization as candidate step 0.
2. Keeps the **best validation-scoring factors**, instead of always exporting final-step factors.
3. Saves both **last optimizer state** and **best factor snapshot** in the resume state. Resume continues from last optimizer step without repeating costly SVD and preserves the previously best factors.
4. Uses separate, independent deterministic **audit probes** for reported initial/final holdout RMSE, so checkpoint selection does not directly use that reported evaluation set.
5. Fails the batch numeric quality gate if the independent audit error regresses, is non-finite, exported packed parity fails, or the normalized RMSE exceeds the threshold.
6. Reports best_step, best_validation_rmse, initial_validation_rmse, last_step_validation_rmse, independent_audit_probes, holdout_error_regressed and the usual actual_effective_bits.

This still uses **random input probes, not actual Gemma hidden-state activations**. Neither a validation nor audit RMSE score establishes real language-model preservation or chat readiness. A checkpoint may be selected for lower validation error and still fail the *independent* audit. Such weights are retained for analysis but fail quality gating and must never be promoted as known-good.

## Rank preview without loading a real matrix

The existing first original Gemma weight shard is enough for this **metadata-only** preview. No new download or training is required:

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"

& $py littlebitchat.py train rank-plan --checkpoint models\gemma-3-27b-it --layer 0 --projection self_attn.q_proj --ranks 32 64 128 256 512 640 --target-bpw 0.55
~~~

For original q_proj shape (4096 × 5376), optional residual factors and current sign/scale accounting:

| Factor rank | Estimated bits / original weight |
|---|---:|
| 32 | 0.04134 |
| 64 | 0.06892 |
| 128 | 0.12407 |
| 256 | 0.23438 |
| 512 | 0.45499 |
| 623 | about 0.551 |
| 640 | 0.56529 |

These values describe compression arithmetic only. Higher rank costs more factor memory, SVD/ITQ work, and training time; it is **not a guaranteed improvement** in language quality. The rank-plan tool does not allocate or train weights and does not claim measured speed/RAM or true serialized checkpoint size.

### Optional bounded comparison, fresh output directory

Only when comfortable with additional CPU compute:

~~~powershell
& $py littlebitchat.py train batch --checkpoint models\gemma-3-27b-it --layer-start 0 --layer-end 1 --projections self_attn.q_proj --rank 128 --steps 50 --device cpu --max-jobs 1 --out training_runs\gemma-q-r128
~~~

**Do not reuse training_runs\gemma-first for rank changes**. The original rank-32 factor artifact remains intact as a failed experiment. The `rank 128` command is exploratory and may still fail the quality gate. The full Gemma 27B checkpoint remains untrained and cannot yet chat.

## Developer verification

Unit tests deliberately inject destructive optimizer updates to prove the trainer exports the previous best validation checkpoint, saves/resumes the best factor state, and uses an independent audit probe set. Rank-plan arithmetic and fail-closed numeric gating are tested separately. The full GitHub Actions suite still runs the prior 160 engine, 12 chat and expanded training tests.

The original dense source is **not** uploaded to GitHub. All trained factors remain under locally ignored training_runs/.
