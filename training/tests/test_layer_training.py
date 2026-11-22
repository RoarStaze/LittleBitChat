import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from training.layer_train import (
    determine_rank, initialize, LittleBitQAT, train_one, save_export,
    packed_cpu_matvec, load_projection, TrainingError,
)


class TrainingTests(unittest.TestCase):
    def test_rank_matches_samsung_target_equation(self):
        self.assertEqual(determine_rank(64, 64, eff_bit=1.0, residual=False, min_rank=8), 8)
        self.assertEqual(determine_rank(64, 64, rank=16), 16)
        with self.assertRaises(ValueError):
            determine_rank(32, 16, rank=17)
        with self.assertRaises(ValueError):
            determine_rank(10, 12, rank=0)

    def test_joint_itq_preserves_unquantized_factor_product(self):
        w = torch.randn(24, 32, generator=torch.Generator().manual_seed(12))
        plain = initialize(w, rank=8, itq_iterations=0, seed=21)
        rotated = initialize(w, rank=8, itq_iterations=7, seed=21)
        torch.testing.assert_close(plain[0] @ plain[1], rotated[0] @ rotated[1], atol=1e-4, rtol=1e-4)

    def test_ste_gradients_reach_binary_factors(self):
        w = torch.randn(16, 24, generator=torch.Generator().manual_seed(3)) * .12
        model = LittleBitQAT.from_teacher(w, rank=8, itq_iterations=4, residual=False, seed=10)
        x = torch.randn(12, 24)
        (model(x) - x @ w.t()).square().mean().backward()
        self.assertTrue(torch.isfinite(model.U.grad).all())
        self.assertGreater(float(model.U.grad.abs().sum()), 0)
        self.assertGreater(float(model.V.grad.abs().sum()), 0)

    def test_export_matches_signed_packed_matvec_with_bias(self):
        w = torch.randn(16, 24, generator=torch.Generator().manual_seed(1)) * .2
        model = LittleBitQAT.from_teacher(w, rank=8, itq_iterations=3, residual=True, seed=11)
        model.eval()
        x = torch.randn(24)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'layer.safetensors'
            prefix = 'language_model.model.layers.1.mlp.down_proj'
            save_export(model, path, prefix)
            observed = packed_cpu_matvec(path, prefix, x.numpy())
            actual = model(x[None]).detach().numpy()[0]
            np.testing.assert_allclose(observed, actual, rtol=1e-5, atol=1e-5)

    def test_training_reduces_holdout_error_and_resumes(self):
        w = torch.randn(12, 20, generator=torch.Generator().manual_seed(7)) * .3
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'state.pt'
            one = train_one(w, rank=8, steps=15, batch_size=32, lr=0.007,
                            itq_iterations=6, seed=8, save_state=state, checkpoint_every=5)
            self.assertTrue(state.exists())
            self.assertEqual(one['step'], 15)
            self.assertTrue(np.isfinite(one['holdout_rmse']))
            self.assertLess(one['holdout_rmse'], one['initial_holdout_rmse'])
            two = train_one(w, rank=8, steps=20, batch_size=32, lr=0.007,
                            itq_iterations=6, seed=8, save_state=state, resume=True, checkpoint_every=5)
            self.assertEqual(two['step'], 20)
            self.assertEqual(two['resumed_from_step'], 15)

    def test_resume_does_not_repeat_expensive_svd_itq(self):
        from unittest.mock import patch
        w = torch.randn(12, 20, generator=torch.Generator().manual_seed(72)) * .3
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'state.pt'
            train_one(w, rank=8, steps=3, batch_size=8, lr=.003,
                      itq_iterations=3, seed=11, save_state=state)
            with patch('training.layer_train.initialize', side_effect=AssertionError('repeated SVD')):
                result = train_one(w, rank=8, steps=5, batch_size=8, lr=.003,
                                   itq_iterations=3, seed=11, save_state=state, resume=True)
            self.assertEqual(result['resumed_from_step'], 3)
            self.assertEqual(result['step'], 5)

    def test_export_is_consumed_by_existing_littlebit_inference_reader(self):
        import sys
        engine = Path(__file__).resolve().parents[2] / 'engine'
        sys.path.insert(0, str(engine))
        from safetensors_layer import SafetensorsLittleBitLayer
        w = torch.randn(16, 24, generator=torch.Generator().manual_seed(91)) * .11
        model = LittleBitQAT.from_teacher(w, rank=8, itq_iterations=4, residual=True, seed=19)
        x = torch.linspace(-1, 1, 24)
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / 'trained.safetensors'
            name = 'language_model.model.layers.0.mlp.up_proj'
            save_export(model, file, name)
            layer = SafetensorsLittleBitLayer.from_file(file, name)
            actual = np.asarray(layer.forward_cpu(x.tolist()), dtype=np.float32)
            expected = model(x[None]).detach().numpy()[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_missing_and_invalid_dense_checkpoint_fails_closed(self):
        with self.assertRaises((TrainingError, FileNotFoundError)):
            load_projection('/does/not/exist', layer=1, projection='mlp.up_proj')
        with self.assertRaises(ValueError):
            load_projection('/does/not/exist', layer=-1, projection='mlp.up_proj')
        with self.assertRaises(ValueError):
            load_projection('/does/not/exist', layer=1, projection='made.up')


if __name__ == '__main__':
    unittest.main()
