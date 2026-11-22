# Resumable LittleBit-2 Gemma Projection Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Extend single-projection LittleBit-2 QAT to offline sequential training of selected original pretrained Gemma 3 27B projections, with safe interrupted-run resume and per-factor quality/integrity records.

**Architecture:** Add a bounded metadata scanner for local dense Safetensors and a sequential training orchestrator that reuses the existing training.cli.run single-layer path. Keep model-wide QAT, trained-weight provenance, full checkpoint assembly and conversational evaluation explicitly out of scope.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, Safetensors, existing Samsung-compatible packed binary factor inference.

**Spec:** docs/BATCH_TRAINING.md

## Global constraints

- No cloud service, paid compute, automatic model download, Qwen substitution or KalshiEdge changes.
- Fail before training if a local dense teacher shard is missing, unsafe, oversized or wrong for the target Gemma 3 27B architecture.
- One dense teacher projection in memory per job; one new job per run by default.
- Completed output and settings must be cryptographically checked before a resumed run skips an existing job.
- Only random-probe normalized RMSE is evaluated; never report language quality as verified.
- All trained outputs remain ignored under training_runs/ and are not a complete chat model.

## Review focus

1. Changing the original teacher after a checkpoint: checksum mismatch prevents reusing stale factors.
2. Changing optimizer/training hyperparameters: reject resume instead of silently mixing experiments.
3. Corrupted packed artifact: reject resume before training any additional projection.
4. Missing/dangerous shard paths: reject metadata inventory without reading external files.
5. Resuming with max-jobs=1: advance to the next untrained projection, not spend the invocation on the first skipped projection.

---

### Task 1: Local metadata scanner

**Files:** training/batch.py, training/tests/test_batch_training.py

- [x] Write a failing inventory test using small real BF16 Safetensors headers.
- [x] Verify expected failure before module creation.
- [x] Implement strict full-size Gemma 3 projection-shape checks and an explicit test-only override.
- [x] Test unsafe/missing sharded references and nonmaterializing scans.

### Task 2: Sequential training and quality records

**Files:** training/batch.py, training/cli.py, training/tests/test_batch_training.py

- [x] Write failing end-to-end tests for two real-format teacher projections.
- [x] Reuse tested SVD/ITQ/SmoothSign single-layer QAT and packed export one job at a time.
- [x] Emit atomic batch manifest with teacher hash, output SHA256, actual BPW and holdout RMSE gate.
- [x] Ensure output never claims full-model training or coherent language.

### Task 3: Durable resume

**Files:** training/batch.py, training/tests/test_batch_training.py

- [x] Test restart from a partially completed training sweep.
- [x] Test changed training settings, modified packed outputs and changed source matrices.
- [x] Confirm a one-job resumption processes a *new* job even if prior jobs are skipped.
- [x] Correct max-jobs semantics after observing red test.

### Task 4: Verification, documentation, merge

**Files:** docs/BATCH_TRAINING.md, README.md, GitHub PR

- [x] Document teacher requirements, quality limitations, exact PowerShell commands.
- [x] Run existing CI with old 160 engine tests, chat contracts and new training tests.
- [ ] Confirm final HEAD CI green after documentation.
- [ ] Review GitHub diff and merge PR to LittleBitChat/main.
- [ ] Provide exact local git pull and training inventory command.

**Follow-up beyond this plan:** Legitimate original pretrained Gemma 3 27B teacher access; activation-aware calibration, model-wide distillation, true full-checkpoint assembly and end-to-end chatbot evaluation. None is achieved by this milestone.
