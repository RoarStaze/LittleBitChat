"""Regression tests for quality protection after a real Gemma projection got worse."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from training.layer_train import train_one, estimated_effective_bits
from training.rank_plan import build_rank_plan

PREFIX = "language_model.model.layers.0.self_attn.q_proj"


class BestCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.w = torch.randn(12, 20, generator=torch.Generator().manual_seed(173)) * .3

    def test_heldout_best_model_survives_destructive_training_steps(self):
        # Force changes which worsen validation to reproduce user-reported
        # last-step regression; the selected best must stay at initialization.
        def poison(optimizer, closure=None):
            with torch.no_grad():
                for param in optimizer.param_groups[0]['params']:
                    if param.ndim == 1:
                        param.mul_(3.0)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "checkpoint.pt"
            with patch.object(torch.optim.AdamW, "step", poison):
                result = train_one(self.w, rank=8, steps=4, batch_size=8,
                    itq_iterations=3, residual=False, seed=81,
                    lr=.002, save_state=state, checkpoint_every=1)
            self.assertEqual(result["best_step"], 0)
            self.assertLessEqual(result["best_validation_rmse"], result["initial_validation_rmse"])
            self.assertGreater(result["last_step_validation_rmse"], result["best_validation_rmse"])
            self.assertAlmostEqual(result["initial_holdout_rmse"], result["holdout_rmse"], places=5)
            self.assertTrue(result["independent_audit_probes"])
            saved = torch.load(state, map_location="cpu", weights_only=True)
            self.assertEqual(saved["best_step"], 0)
            self.assertEqual(saved["step"], 4)
            self.assertIn("best_model", saved)
            self.assertIn("initial_holdout_rmse", saved)
            self.assertIn("initial_validation_rmse", saved)

    def test_resume_preserves_best_factor_snapshot_without_rerunning_svd(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "checkpoint.pt"
            one = train_one(self.w, rank=8, steps=4, batch_size=8,
                itq_iterations=2, residual=False, seed=88, save_state=state,
                checkpoint_every=2)
            from unittest.mock import patch as patched
            with patched("training.layer_train.initialize", side_effect=AssertionError("SVD reran")):
                two = train_one(self.w, rank=8, steps=7, batch_size=8,
                    itq_iterations=2, residual=False, seed=88, save_state=state,
                    checkpoint_every=2, resume=True)
            self.assertEqual(two["resumed_from_step"], 4)
            self.assertLessEqual(two["best_validation_rmse"], one["best_validation_rmse"] + 1e-7)
            self.assertEqual(two["initial_holdout_rmse"], one["initial_holdout_rmse"])

    def test_train_one_reports_selection_vs_unseen_audit(self):
        result = train_one(self.w, rank=8, steps=3, batch_size=6,
            residual=False, seed=22, itq_iterations=2, checkpoint_every=1)
        self.assertTrue(result["independent_audit_probes"])
        self.assertLessEqual(result["best_validation_rmse"], result["initial_validation_rmse"] + 1e-7)
        self.assertIn("last_step_validation_rmse", result)
        self.assertIn("best_step", result)
        self.assertTrue(0 <= result["best_step"] <= result["step"])


class RankPlanTests(unittest.TestCase):
    def test_actual_gemma_dimensions_and_target_bpw_are_calculated_from_header(self):
        # Actual tensor need not be downloaded to test arithmetic; a metadata
        # source mock proves the plan's dimensions and low-rank warning.
        info = [{"layer":0,"projection":"self_attn.q_proj",
                 "shape":[4096, 5376], "dtype":"BF16",
                 "source_shard":"model-00001.safetensors",
                 "matches_gemma3_27b_shape":True}]
        with patch("training.rank_plan.scan_source", return_value={
                    "projections": info, "available_projections":1}):
            result = build_rank_plan("not-read", layer=0, projection="self_attn.q_proj",
                                     ranks=(32,64,128,256), target_bpw=0.55)
        self.assertEqual(result["status"], "RANK_BUDGET_PLAN_ONLY")
        self.assertAlmostEqual(result["options"][0]["estimated_effective_bpw"],
                               0.041341145833333336, places=12)
        self.assertGreater(result["rank_for_target_bpw"], 256)
        self.assertGreaterEqual(result["estimated_bpw_at_target_rank"], 0.50)
        self.assertFalse(result["predicts_model_quality"])
        self.assertFalse(result["trains_or_allocates_weights"])
        self.assertFalse(result["trains_full_model"])

    def test_invalid_ranks_target_bpw_and_dims_fail_fast(self):
        fake = [{"layer":0,"projection":"self_attn.q_proj","shape":[24,32],
                 "dtype":"BF16","matches_gemma3_27b_shape":False}]
        with patch("training.rank_plan.scan_source", return_value={"projections": fake}):
            with self.assertRaises(ValueError):
                build_rank_plan("x",layer=0,projection="self_attn.q_proj",ranks=(0,))
            with self.assertRaises(ValueError):
                build_rank_plan("x",layer=0,projection="self_attn.q_proj",ranks=(33,))
            with self.assertRaises(ValueError):
                build_rank_plan("x",layer=0,projection="self_attn.q_proj",target_bpw=float("nan"))
            good=build_rank_plan("x",layer=0,projection="self_attn.q_proj",ranks=(8,),
                                 allow_test_shapes=True)
        self.assertEqual(good["options"][0]["rank"],8)

    def test_rank_plan_cli_exists_and_does_not_run_training(self):
        import subprocess,sys
        root=Path(__file__).resolve().parents[2]
        cmd=[sys.executable,str(root/"littlebitchat.py"),"train","rank-plan","--help"]
        proc=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=30)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertIn("--ranks",proc.stdout)
        self.assertIn("--target-bpw",proc.stdout)


if __name__=="__main__":
    unittest.main()
