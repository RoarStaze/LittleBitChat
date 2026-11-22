"""Accuracy guards for oversampled randomized SVD and multi-pass binary refinement."""
import tempfile
import unittest
from pathlib import Path

import torch

from training.binary_refinement import refine_binary_signs
from training.diagnose import diagnose_matrix
from training.layer_train import LittleBitQAT, TrainingError, initialize, train_one
from training.tests.test_binary_refinement import fixture


class StrongerSVDBaselineTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(120)
        self.teacher = torch.randn(24, 36, generator=gen)

    def test_oversampled_power_iteration_never_worsens_rank_limited_weight_error(self):
        base = initialize(self.teacher, rank=6, itq_iterations=0, seed=33)
        improved = initialize(self.teacher, rank=6, itq_iterations=0, seed=33,
                              svd_oversample=8, svd_power_iters=5)
        base_sse = float((self.teacher - base[0] @ base[1]).square().sum())
        improved_sse = float((self.teacher - improved[0] @ improved[1]).square().sum())
        self.assertLessEqual(improved_sse, base_sse + 1e-5)
        self.assertEqual(tuple(improved[0].shape), (24, 6))
        self.assertEqual(tuple(improved[1].shape), (6, 36))

    def test_zero_oversampling_and_three_iterations_preserve_old_behavior(self):
        old = initialize(self.teacher, rank=5, seed=11, itq_iterations=3)
        explicit = initialize(self.teacher, rank=5, seed=11, itq_iterations=3,
                              svd_oversample=0, svd_power_iters=3)
        torch.testing.assert_close(old[0], explicit[0], rtol=0, atol=0)
        torch.testing.assert_close(old[1], explicit[1], rtol=0, atol=0)

    def test_full_rank_clamps_oversampling_without_dimension_errors(self):
        u, v = initialize(self.teacher, rank=24, seed=4, itq_iterations=0,
                          svd_oversample=16, svd_power_iters=4)
        self.assertEqual(tuple(u.shape), (24, 24))
        self.assertEqual(tuple(v.shape), (24, 36))

    def test_bad_svd_configuration_fails(self):
        for oversample in (-1, 1.2, True, 257):
            with self.subTest(svd_oversample=oversample), self.assertRaises(ValueError):
                initialize(self.teacher, rank=6, itq_iterations=0,
                           svd_oversample=oversample)
        for iters in (-1, True, 11):
            with self.subTest(svd_power_iters=iters), self.assertRaises(ValueError):
                initialize(self.teacher, rank=6, itq_iterations=0,
                           svd_power_iters=iters)


class MultiPassBinaryTests(unittest.TestCase):
    def test_extra_passes_preserve_or_improve_sse_on_same_factorization(self):
        a = fixture(residual=True)
        b = fixture(residual=True)
        teacher = torch.randn(8, 12, generator=torch.Generator().manual_seed(2))
        single = refine_binary_signs(a, teacher, channels=5, passes=1)
        multi = refine_binary_signs(b, teacher, channels=5, passes=3)
        self.assertLessEqual(multi["weight_sse_after"],
                             single["weight_sse_after"] + 1e-4)
        self.assertLessEqual(multi["weight_sse_after"], multi["weight_sse_before"] + 1e-4)
        self.assertEqual(multi["passes_requested"], 3)
        self.assertTrue(1 <= multi["passes_completed"] <= 3)
        self.assertEqual(multi["examined_channels"], 5)
        self.assertGreaterEqual(multi["accepted_updates"], multi["accepted_channels"])

    def test_one_pass_keeps_original_result(self):
        a, b = fixture(), fixture()
        teacher = torch.randn(8, 12, generator=torch.Generator().manual_seed(21))
        old = refine_binary_signs(a, teacher, channels=3)
        new = refine_binary_signs(b, teacher, channels=3, passes=1)
        torch.testing.assert_close(a.dense_weight(), b.dense_weight(), rtol=0, atol=0)
        self.assertEqual(old["weight_sse_after"], new["weight_sse_after"])

    def test_invalid_pass_limit(self):
        for n in (0, 9, -1, True, 1.5):
            with self.subTest(passes=n), self.assertRaises(ValueError):
                refine_binary_signs(fixture(), torch.zeros(8, 12),
                                    channels=2, passes=n)


class IntegrationTests(unittest.TestCase):
    def test_diagnose_discloses_legacy_and_enhanced_separately(self):
        w = torch.randn(12, 16, generator=torch.Generator().manual_seed(11))
        report = diagnose_matrix(w, rank=3, seed=12, itq_iterations=2,
                                 residual=True, probe_count=12,
                                 refine_output_scales=True,
                                 binary_refine_channels=3, binary_refine_passes=3,
                                 svd_oversample=4, svd_power_iters=5)
        self.assertLessEqual(report["svd_enhanced_relative_weight_rmse"],
                             report["svd_relative_weight_rmse"] + 1e-5)
        self.assertIsNotNone(report["svd_enhanced_random_probe_nrmse"])
        self.assertIsNotNone(report["binary_svd_enhanced_relative_weight_rmse"])
        self.assertEqual(report["binary_refine_passes"], 3)
        self.assertEqual(report["binary_refine_details"]["passes_requested"], 3)
        self.assertEqual(report["svd_oversample"], 4)
        self.assertEqual(report["svd_power_iters"], 5)
        self.assertFalse(report["training_was_run"])
        self.assertEqual(report["estimated_effective_bpw"], report["quantization_bpw"])

    def test_training_resume_rejects_changed_foundation_settings(self):
        w = torch.randn(8, 12, generator=torch.Generator().manual_seed(8))
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / "saved.pt"
            one = train_one(w, rank=3, steps=1, batch_size=4,
                            residual=False, itq_iterations=1,
                            binary_refine_channels=2, binary_refine_passes=2,
                            svd_oversample=2, svd_power_iters=4,
                            save_state=state)
            self.assertEqual(one["svd_oversample"], 2)
            self.assertEqual(one["binary_refine_passes"], 2)
            with self.assertRaisesRegex(TrainingError, "mismatch"):
                train_one(w, rank=3, steps=2, batch_size=4,
                          residual=False, itq_iterations=1,
                          binary_refine_channels=2, binary_refine_passes=1,
                          svd_oversample=2, svd_power_iters=4,
                          save_state=state, resume=True)
            two = train_one(w, rank=3, steps=2, batch_size=4,
                            residual=False, itq_iterations=1,
                            binary_refine_channels=2, binary_refine_passes=2,
                            svd_oversample=2, svd_power_iters=4,
                            save_state=state, resume=True)
            self.assertEqual(two["resumed_from_step"], 1)


if __name__ == "__main__":
    unittest.main()
