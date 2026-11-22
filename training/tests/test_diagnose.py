"""Tests the separation between low rank energy loss and sign quantization loss."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from training.diagnose import diagnose_matrix, diagnose_projection, DiagnosisError

ROOT=Path(__file__).resolve().parents[2]
KEY="language_model.model.layers.0.self_attn.q_proj.weight"


class QuantizationDiagnosisTests(unittest.TestCase):
    def setUp(self):
        gen=torch.Generator().manual_seed(19)
        self.matrix=(torch.randn(12,3,generator=gen) @
                     torch.randn(3,16,generator=gen)).float()

    def test_lowrank_teacher_svd_retains_rank3_matrix_but_sign_loss_is_visible(self):
        report=diagnose_matrix(self.matrix,rank=3,seed=42,itq_iterations=2,
                               residual=False,probe_count=16)
        self.assertEqual(report["status"],"PROJECTION_APPROXIMATION_DIAGNOSIS")
        self.assertLess(report["svd_relative_weight_rmse"], 0.001)
        self.assertGreater(report["binary_initial_relative_weight_rmse"],
                           report["svd_relative_weight_rmse"] + 0.01)
        self.assertTrue(report["lowrank_approximation_is_randomized"])
        self.assertEqual(report["rank"],3)
        self.assertEqual(report["teacher_shape"],[12,16])
        self.assertEqual(report["quantization_bpw"], report["estimated_effective_bpw"])
        self.assertEqual(report["probe_count"],16)
        self.assertFalse(report["measures_language_quality"])
        self.assertFalse(report["full_model_trained"])
        self.assertFalse(report["training_was_run"])

    def test_rank_is_validated_before_expensive_initialization(self):
        for invalid in (0,13,-1,1.5):
            with self.subTest(rank=invalid),self.assertRaises((ValueError,DiagnosisError)):
                diagnose_matrix(self.matrix,rank=invalid)
        with self.assertRaises((ValueError,DiagnosisError)):
            diagnose_matrix(self.matrix,rank=3,probe_count=0)
        with self.assertRaises((ValueError,DiagnosisError)):
            diagnose_matrix(torch.full((12,16),float("nan")),rank=3)

    def test_binary_refinement_reports_baseline_and_is_weight_nonregressing(self):
        baseline = diagnose_matrix(self.matrix, rank=3, seed=42,
                                   itq_iterations=2, probe_count=16)
        refined = diagnose_matrix(self.matrix, rank=3, seed=42,
                                  itq_iterations=2, probe_count=16,
                                  refine_output_scales=True)
        self.assertEqual(refined["teacher_sha256"], baseline["teacher_sha256"])
        self.assertAlmostEqual(refined["binary_initial_relative_weight_rmse"],
                               baseline["binary_initial_relative_weight_rmse"], places=6)
        self.assertLessEqual(refined["binary_refined_relative_weight_rmse"],
                             baseline["binary_initial_relative_weight_rmse"] + 1e-6)
        self.assertLessEqual(refined["refinement_weight_sse_after"],
                             refined["refinement_weight_sse_before"] + 1e-6)
        self.assertTrue(refined["refine_output_scales"])
        self.assertIsNotNone(refined["binary_refined_random_probe_nrmse"])
        self.assertFalse(refined["training_was_run"])
        self.assertEqual(refined["quantization_bpw"], baseline["quantization_bpw"])

    def test_greedy_binary_sign_diagnostic_keeps_baseline_and_bpw(self):
        baseline = diagnose_matrix(self.matrix, rank=3, seed=42,
                                   itq_iterations=2, probe_count=16)
        improved = diagnose_matrix(self.matrix, rank=3, seed=42,
                                   itq_iterations=2, probe_count=16,
                                   refine_output_scales=True,
                                   binary_refine_channels=3)
        self.assertEqual(improved["teacher_sha256"], baseline["teacher_sha256"])
        self.assertAlmostEqual(improved["binary_initial_relative_weight_rmse"],
                               baseline["binary_initial_relative_weight_rmse"], places=6)
        self.assertLessEqual(improved["binary_binary_refined_relative_weight_rmse"],
                             improved["binary_refined_relative_weight_rmse"] + 1e-6)
        self.assertEqual(improved["quantization_bpw"], baseline["quantization_bpw"])
        self.assertEqual(improved["binary_refine_channels"], 3)
        self.assertEqual(improved["binary_refine_details"]["examined_channels"], 3)
        self.assertIsNotNone(improved["binary_binary_refined_random_probe_nrmse"])
        self.assertFalse(improved["training_was_run"])

    def test_refinement_option_must_be_boolean(self):
        with self.assertRaises(DiagnosisError):
            diagnose_matrix(self.matrix, rank=3, refine_output_scales="yes")

    def test_missing_real_teacher_fails_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"absent.safetensors"
            with self.assertRaisesRegex((ValueError,DiagnosisError),"checkpoint|not found"):
                diagnose_projection(p,layer=0,projection="self_attn.q_proj",rank=3)
            self.assertFalse(p.exists())

    def test_real_format_bf16_checkpoint_diagnosis_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"original.safetensors"
            save_file({KEY:self.matrix.bfloat16()},str(source))
            before=source.read_bytes()
            report=diagnose_projection(source,layer=0,
                                       projection="self_attn.q_proj",
                                       rank=3,itq_iterations=2,probe_count=12)
            self.assertFalse(report["synthetic_teacher_data"])
            self.assertEqual(report["teacher_sha256"],report["teacher_sha256"].lower())
            self.assertEqual(len(report["teacher_sha256"]),64)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),["original.safetensors"])

    def test_command_line_runs_with_real_format_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"original.safetensors"
            save_file({KEY:self.matrix.bfloat16()},str(source))
            args=[sys.executable,str(ROOT/"littlebitchat.py"),"train","diagnose",
                "--checkpoint",str(source),"--layer","0",
                "--projection","self_attn.q_proj","--rank","3",
                "--itq-iterations","2","--samples","12","--no-residual",
                "--refine-output-scales"]
            p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(p.returncode,0,p.stderr)
            report=json.loads(p.stdout)
            self.assertEqual(report["status"],"PROJECTION_APPROXIMATION_DIAGNOSIS")
            self.assertFalse(report["training_was_run"])
            self.assertTrue(report["refine_output_scales"])
            self.assertIsNotNone(report["binary_refined_relative_weight_rmse"])


if __name__=="__main__":
    unittest.main()
