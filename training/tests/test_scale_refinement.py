"""Output-row calibration must improve a fixed binary factorization without extra bits."""
import unittest
import tempfile
from pathlib import Path

import torch

from training.layer_train import (LittleBitQAT, TrainingError, estimated_effective_bits,
                                  train_one)
from training.scale_refinement import refine_output_row_scales


def toy_model(residual=True, collinear=False):
    gen = torch.Generator().manual_seed(782)
    rows, cols, rank = 5, 7, 4
    components = {
        "U": torch.randn(rows, rank, generator=gen),
        "V": torch.randn(rank, cols, generator=gen),
        "u1": torch.ones(rows),
        "u2": torch.ones(rank),
        "v1": torch.ones(rank),
        "v2": torch.ones(cols),
    }
    if residual:
        for key in ("U", "V", "u1", "u2", "v1", "v2"):
            if collinear:
                components[key + "_R"] = components[key].clone()
            elif key in ("U", "V"):
                shape = components[key].shape
                components[key + "_R"] = torch.randn(shape, generator=gen)
            else:
                components[key + "_R"] = torch.ones_like(components[key])
    return LittleBitQAT(components)


class ScaleRefinementTests(unittest.TestCase):
    def test_joint_nonnegative_calibration_recovers_two_path_teacher(self):
        model = toy_model()
        primary = model.dense_path_weight("")
        residual = model.dense_path_weight("_R")
        teacher = (primary * torch.tensor([0.5, 1.1, 1.7, 2.0, 0.8])[:, None] +
                   residual * torch.tensor([2.0, 1.4, 0.9, 0.3, 1.5])[:, None])
        result = refine_output_row_scales(model, teacher)
        self.assertTrue(result["changed"])
        self.assertLess(result["weight_sse_after"], 1e-7)
        self.assertLess(result["weight_sse_after"], result["weight_sse_before"])
        self.assertTrue(torch.all(model.u1 >= 0))
        self.assertTrue(torch.all(model.u1_R >= 0))
        x = torch.arange(7, dtype=torch.float32)
        torch.testing.assert_close(model(x), model.dense_weight() @ x, atol=1e-5, rtol=1e-5)
        self.assertEqual(estimated_effective_bits(5, 7, 4, True),
                         estimated_effective_bits(5, 7, 4, model.residual))

    def test_one_path_improves_without_residual(self):
        model = toy_model(residual=False)
        original = model.dense_weight()
        teacher = original * torch.tensor([2.0, 1.5, 0.2, 0.6, 1.7])[:, None]
        result = refine_output_row_scales(model, teacher)
        self.assertLess(result["weight_sse_after"], 1e-7)
        self.assertFalse(model.residual)

    def test_collinear_factors_are_stable(self):
        model = toy_model(collinear=True)
        teacher = model.dense_weight() * 0.7
        result = refine_output_row_scales(model, teacher)
        self.assertLessEqual(result["weight_sse_after"], result["weight_sse_before"] + 1e-8)
        self.assertTrue(torch.isfinite(model.dense_weight()).all())

    def test_training_resume_rejects_changed_refinement_policy(self):
        gen = torch.Generator().manual_seed(22)
        teacher = torch.randn(5, 7, generator=gen)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "state.pt"
            one = train_one(teacher, rank=2, steps=1, batch_size=3,
                            itq_iterations=0, residual=False,
                            save_state=checkpoint, refine_output_scales=True)
            self.assertTrue(one["refine_output_scales"])
            with self.assertRaisesRegex(TrainingError, "mismatch"):
                train_one(teacher, rank=2, steps=2, batch_size=3,
                          itq_iterations=0, residual=False,
                          save_state=checkpoint, resume=True,
                          refine_output_scales=False)
            two = train_one(teacher, rank=2, steps=2, batch_size=3,
                            itq_iterations=0, residual=False,
                            save_state=checkpoint, resume=True,
                            refine_output_scales=True)
            self.assertEqual(two["resumed_from_step"], 1)

    def test_rejects_teacher_shape_mismatch(self):
        model = toy_model()
        with self.assertRaises(ValueError):
            refine_output_row_scales(model, torch.zeros(4, 7))
        with self.assertRaises(ValueError):
            refine_output_row_scales(model, torch.full((5, 7), float("nan")))


if __name__ == "__main__":
    unittest.main()
