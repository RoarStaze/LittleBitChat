"""Tests for same-bit-budget binary sign coordinate descent."""
import tempfile
import unittest
from pathlib import Path

import torch

from training.binary_refinement import refine_binary_signs
from training.layer_train import LittleBitQAT, TrainingError, estimated_effective_bits, train_one


def fixture(*, residual=False):
    g = torch.Generator().manual_seed(27)
    rows, cols, rank = 8, 12, 3
    factors = dict(U=torch.randn(rows, rank, generator=g),
                   V=torch.randn(rank, cols, generator=g),
                   u1=torch.linspace(.2, 1.8, rows), u2=torch.tensor([1.2, .8, .9]),
                   v1=torch.tensor([2.0, .6, .7]), v2=torch.linspace(.3, 1.5, cols))
    if residual:
        factors.update({key+"_R": (torch.randn(value.shape, generator=g)
                                   if key in ("U", "V") else value.clone())
                        for key, value in factors.items()})
    return LittleBitQAT(factors)


class BinaryRefinementTests(unittest.TestCase):
    def test_finds_real_sign_flips_without_increasing_bits(self):
        model = fixture()
        desired = fixture()
        with torch.no_grad():
            desired.U[:4, 0].mul_(-1)
            desired.V[0, :4].mul_(-1)
        teacher = desired.dense_weight()
        before = float((model.dense_weight()-teacher).square().sum())
        u_mag = model.U.abs().clone()
        v_mag = model.V.abs().clone()
        bits = estimated_effective_bits(8, 12, 3, False)
        result = refine_binary_signs(model, teacher, channels=3)
        self.assertLess(result["weight_sse_after"], before - 1e-4)
        self.assertLess(result["weight_sse_after"], 1e-7)
        self.assertGreater(result["accepted_channels"], 0)
        self.assertTrue(result["changed"])
        torch.testing.assert_close(model.U.abs(), u_mag)
        torch.testing.assert_close(model.V.abs(), v_mag)
        self.assertEqual(bits, estimated_effective_bits(8, 12, 3, model.residual))
        x = torch.randn(4, 12, generator=torch.Generator().manual_seed(6))
        torch.testing.assert_close(model(x), x@model.dense_weight().T, atol=1e-5, rtol=1e-5)

    def test_residual_paths_and_one_channel_never_regress(self):
        g = torch.Generator().manual_seed(234)
        model = fixture(residual=True)
        teacher = torch.randn(8, 12, generator=g)
        report = refine_binary_signs(model, teacher, channels=1)
        self.assertLessEqual(report["weight_sse_after"],
                             report["weight_sse_before"] + 1e-5)
        self.assertEqual(report["examined_channels"], 1)
        self.assertTrue(torch.isfinite(model.dense_weight()).all())

    def test_zero_budget_is_unchanged(self):
        model = fixture()
        before = model.dense_weight().clone()
        out = refine_binary_signs(model, before, channels=0)
        self.assertFalse(out["changed"])
        torch.testing.assert_close(model.dense_weight(), before)

    def test_invalid_inputs_fail_before_mutation(self):
        model = fixture()
        saved = model.dense_weight().clone()
        for invalid in (-1, 1.5, True, 257):
            with self.subTest(channels=invalid), self.assertRaises(ValueError):
                refine_binary_signs(model, saved, channels=invalid)
        for teacher in (torch.zeros(7, 12),
                        torch.full((8, 12), float("nan"))):
            with self.assertRaises(ValueError):
                refine_binary_signs(model, teacher, channels=2)
        torch.testing.assert_close(model.dense_weight(), saved)

    def test_train_resume_checks_binary_refinement_policy(self):
        teacher = torch.randn(8, 12, generator=torch.Generator().manual_seed(9))
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.pt"
            first = train_one(teacher, rank=3, steps=1, batch_size=4,
                              itq_iterations=1, residual=False,
                              binary_refine_channels=2, save_state=state)
            self.assertEqual(first["binary_refine_channels"], 2)
            with self.assertRaisesRegex(TrainingError, "mismatch"):
                train_one(teacher, rank=3, steps=2, batch_size=4,
                          itq_iterations=1, residual=False,
                          binary_refine_channels=0, resume=True, save_state=state)
            second = train_one(teacher, rank=3, steps=2, batch_size=4,
                               itq_iterations=1, residual=False,
                               binary_refine_channels=2, resume=True, save_state=state)
            self.assertEqual(second["resumed_from_step"], 1)


if __name__ == "__main__":
    unittest.main()
