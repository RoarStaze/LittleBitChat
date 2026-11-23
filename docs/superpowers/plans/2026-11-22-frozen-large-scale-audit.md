# Large-Scale Token-Disjoint Validation with Frozen Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Obtain a larger, reproducible, token-disjoint audit of Gemma 3 layer-0 Q-projection compression on the user's 15.7-GB-RAM laptop, without the 27B model, and freeze the independent audit against later experiment drift.

**Architecture:** Existing `training/layer0_capture.py` provides original-teacher binding and exact sparse embedding+RMSNorm reconstruction; existing `activation_calibration.py` provides disjoint-array validation, selection-only accept/reject and unexamined audit. A new `training/frozen_audit.py` owns deterministic vocabulary/corpus token sampling, audit lock, sparse capture, and paired bootstrap uncertainty. `training/cli.py` exposes opt-in capture and audit-lock checks; defaults and legacy commands remain unchanged. All source reads local-only. Explicitly label vocabulary-uniform/token-type benchmarking as distinct from natural-language distribution.

## Tasks

- [ ] DYX-001: Add RED tests in `training/tests/test_frozen_audit.py` for reproducible 2048/1024/1024-style disjoint sampling with tiny fixtures; special token exclusion; fixed audit across seed changes; no lock overwrite; corrupted lock/capture failure; teacher mismatch; audit source replay.
- [ ] DYX-002: Implement `training/frozen_audit.py` with deterministic sampling, input validation, strict file I/O, initial freeze/reuse lock and exact replay-compatible Safetensors capture; rerun red tests to GREEN.
- [ ] DYX-003: Add RED tests for CLI `capture-vocabulary`, strict `activation-audit --frozen-audit-lock`, `activation-calibrate --frozen-audit-lock`, failure before expensive SVD and report lock SHA; implement CLI integrations.
- [ ] DYX-004: Add RED test and deterministic paired per-token bootstrap confidence interval on independent frozen audit, computed **after** selection acceptance; include uncertainty disclaimer; implement and test.
- [ ] DYX-005: Document commands, locked artifact custody, source-distribution limitations, new outputs and failure modes in `docs/FROZEN_TOKEN_AUDIT.md` and README.
- [ ] DYX-006: Run GitHub CI test suite at exact final commit; review evidence and changed files; merge PR to main; provide precise PowerShell pull/capture/audit/calibration commands.

## Invariants

1. Selected token IDs are unique across calibration/selection/audit, and token-bearing arrays match original local weight rows on strict replay.
2. Audit IDs and activation bytes are recorded in a never-overwritten, SHA-256-bound lock; changing the calibration sample seed does not change frozen audit.
3. A new lock is created only when absent; a reused lock must pass binding and equality checks before new captures.
4. Calibration's acceptance uses **only selection data**. Audit is observed after the choice; confidence interval is descriptive, never an optimization gate.
5. Estimated BPW and sign factors unchanged by audit features. No network access, 27B model loading, or language-quality claims.
6. The original limited text-based capture and strict prompt-disjoint workflows continue to operate.
