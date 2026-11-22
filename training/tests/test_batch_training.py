"""Batch LittleBit-2 training: real dense source, safety gates, durable resume.

The small BF16 fixture exercises file-backed training, not pretrained Gemma
knowledge or anything resembling complete 27B conversational quality.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from training.batch import scan_source, run_batch, BatchTrainingError

ROOT = Path(__file__).resolve().parents[2]
BASE = "language_model.model.layers.0."
PROJS = ("self_attn.q_proj", "self_attn.k_proj")


class BatchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "teacher"
        self.source.mkdir()
        self.out = self.root / "learned"
        self.weights = {
            BASE + "self_attn.q_proj.weight": torch.randn(16, 24, generator=torch.Generator().manual_seed(3)).bfloat16(),
            BASE + "self_attn.k_proj.weight": torch.randn(8, 24, generator=torch.Generator().manual_seed(4)).bfloat16(),
        }
        save_file(self.weights, str(self.source / "model-00001.safetensors"))
        mapping = {key: "model-00001.safetensors" for key in self.weights}
        (self.source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))

    def _kwargs(self, **changes):
        args = dict(source=self.source, out=self.out, layer_start=0, layer_end=1,
            projections=PROJS, rank=8, steps=5, batch_size=8,
            lr=0.003, itq_iterations=2, seed=11, device="cpu",
            residual=False, max_teacher_mib=16, checkpoint_every=5,
            max_jobs=2, max_holdout_nrmse=0.25, allow_test_shapes=True,
            resume=False)
        args.update(changes)
        return args

    def test_inventory_is_metadata_only_and_does_not_create_output(self):
        report = scan_source(self.source, layer_start=0, layer_end=1,
                             projections=PROJS, allow_test_shapes=True)
        self.assertEqual(report["status"], "DENSE_TEACHER_PARTIAL")
        self.assertEqual(report["requested_projections"], 2)
        self.assertEqual(report["available_projections"], 2)
        self.assertFalse(report["full_model_trained"])
        self.assertFalse(report["trained_littlebit_weights_present"])
        self.assertFalse(self.out.exists())
        self.assertEqual(report["projections"][0]["shape"], [16, 24])

    def test_default_real_gemma_shape_gate_rejects_fake_weights(self):
        with self.assertRaisesRegex(BatchTrainingError, "27B|shape"):
            scan_source(self.source, layer_start=0, layer_end=1, projections=PROJS)
        self.assertFalse(self.out.exists())

    def test_training_two_real_safetensors_tensors_and_manifest(self):
        report = run_batch(**self._kwargs())
        self.assertEqual(report["status"], "BATCH_LAYER_QAT_PARTIAL")
        self.assertEqual(report["completed_jobs"], 2)
        self.assertEqual(report["total_jobs"], 2)
        self.assertEqual(report["skipped_jobs"], 0)
        self.assertFalse(report["full_model_trained"])
        self.assertFalse(report["ready_for_chat"])
        self.assertFalse(report["validated_language_quality"])
        self.assertTrue((self.out / "batch_manifest.json").exists())
        for job in report["jobs"]:
            self.assertTrue(Path(job["packed_file"]).exists())
            self.assertTrue(job["packed_inference_verified"])
            self.assertTrue(job["teacher_weights_loaded_from_real_checkpoint"])
            self.assertIn("passes_random_probe_gate", job)
            self.assertTrue(job["packed_sha256"])
            self.assertEqual(len(job["packed_sha256"]), 64)
        self.assertEqual(len(report["jobs"]), 2)

    def test_resume_skips_verified_completed_exports(self):
        a = run_batch(**self._kwargs())
        paths = [Path(j["packed_file"]) for j in a["jobs"]]
        checksums = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        b = run_batch(**self._kwargs(resume=True))
        self.assertEqual(b["skipped_jobs"], 2)
        self.assertEqual(b["completed_jobs"], 2)
        self.assertEqual(checksums, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])

    def test_partial_batch_resumes_without_retraining_first_projection(self):
        first = run_batch(**self._kwargs(max_jobs=1))
        self.assertEqual(first["completed_jobs"], 1)
        second = run_batch(**self._kwargs(resume=True, max_jobs=2))
        self.assertEqual(second["completed_jobs"], 2)
        self.assertEqual(second["skipped_jobs"], 1)
        self.assertEqual(second["new_jobs"], 1)

    def test_resume_detects_corrupted_packed_tensor(self):
        run_batch(**self._kwargs(max_jobs=1))
        artifact = next(self.out.glob("*.safetensors"))
        with artifact.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(BatchTrainingError, "checksum|modified|corrupt"):
            run_batch(**self._kwargs(resume=True, max_jobs=2))

    def test_resume_refuses_changed_hyperparameters(self):
        run_batch(**self._kwargs(max_jobs=1))
        with self.assertRaisesRegex(BatchTrainingError, "configuration|mismatch"):
            run_batch(**self._kwargs(resume=True, lr=0.004))

    def test_bad_manifest_cannot_escape_checkpoint_directory(self):
        index = self.source / "model.safetensors.index.json"
        data = json.loads(index.read_text())
        data["weight_map"][BASE + "self_attn.q_proj.weight"] = "../escape.safetensors"
        index.write_text(json.dumps(data))
        with self.assertRaisesRegex(BatchTrainingError, "Unsafe|unsafe"):
            scan_source(self.source, layer_start=0, layer_end=1,
                        projections=PROJS, allow_test_shapes=True)

    def test_missing_second_projection_is_a_hard_error(self):
        index = self.source / "model.safetensors.index.json"
        data = json.loads(index.read_text())
        del data["weight_map"][BASE + "self_attn.k_proj.weight"]
        index.write_text(json.dumps(data))
        with self.assertRaisesRegex(BatchTrainingError, "missing|Missing"):
            scan_source(self.source, layer_start=0, layer_end=1,
                        projections=PROJS, allow_test_shapes=True)

    def test_command_line_inventory_and_batch_are_available(self):
        args = [sys.executable, str(ROOT / "littlebitchat.py"), "train", "inventory",
                "--checkpoint", str(self.source), "--layer-start", "0", "--layer-end", "1",
                "--projections", *PROJS, "--allow-test-shapes"]
        r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=45)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["available_projections"], 2)
        self.assertNotIn("training_runs", r.stdout)

    def test_invalid_quality_gate_fails_before_any_training(self):
        with self.assertRaisesRegex(BatchTrainingError, "quality|nrmse"):
            run_batch(**self._kwargs(max_holdout_nrmse=float("nan")))
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
