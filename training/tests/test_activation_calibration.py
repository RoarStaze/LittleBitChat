"""Safety and efficacy tests for same-BPW activation-aware post-training calibration."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from training.activation_calibration import (
    ActivationDataError, calibrate_activation_scales, load_activation_bundle,
)
from training.layer_train import LittleBitQAT, _teacher_sha, estimated_effective_bits
from training.tests.test_binary_refinement import fixture


def fixture_bundle(path, teacher, calibration, selection, audit, *, layer=0,
                   projection="self_attn.q_proj", **changes):
    metadata = dict(
        teacher_sha256=_teacher_sha(teacher),
        layer=str(layer), projection=projection,
        capture_method="forward_pre_hook",
        split_method="disjoint_prompts",
        calibration_text_sha256="a"*64,
        selection_text_sha256="b"*64,
        audit_text_sha256="c"*64,
        input_module=f"language_model.model.layers.{layer}.{projection}",
        source_model="local-user-supplied-original-checkpoint",
    )
    metadata.update(changes)
    save_file({"calibration":calibration.contiguous(),
               "selection":selection.contiguous(),
               "audit":audit.contiguous()}, str(path), metadata=metadata)


class ActivationBundleTests(unittest.TestCase):
    def setUp(self):
        self.teacher = torch.randn(8, 12, generator=torch.Generator().manual_seed(41))
        g = torch.Generator().manual_seed(39)
        self.groups = [torch.randn(20, 12, generator=g) for _ in range(3)]

    def test_bundle_requires_source_binding_and_three_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"capture.safetensors"
            fixture_bundle(path, self.teacher, *self.groups)
            loaded=load_activation_bundle(path, self.teacher, layer=0,
                                          projection="self_attn.q_proj")
            self.assertEqual(tuple(loaded.calibration.shape), (20,12))
            self.assertEqual(tuple(loaded.selection.shape), (20,12))
            self.assertEqual(tuple(loaded.audit.shape), (20,12))
            self.assertFalse(loaded.provenance_independently_verified)
            self.assertEqual(len(loaded.activation_sha256), 64)
            self.assertEqual(loaded.metadata["capture_method"], "forward_pre_hook")

    def test_rejects_wrong_checkpoint_and_wrong_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.safetensors"
            fixture_bundle(path, self.teacher, *self.groups)
            with self.assertRaisesRegex(ActivationDataError, "teacher|SHA"):
                load_activation_bundle(path, self.teacher+0.1, layer=0,
                                       projection="self_attn.q_proj")
            with self.assertRaisesRegex(ActivationDataError, "projection|module"):
                load_activation_bundle(path, self.teacher, layer=0,
                                       projection="self_attn.k_proj")

    def test_rejects_overlap_duplicate_or_nonfinite_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.safetensors"
            fixture_bundle(path, self.teacher, self.groups[0],
                           self.groups[0].clone(), self.groups[2])
            with self.assertRaisesRegex(ActivationDataError, "overlap|duplicate"):
                load_activation_bundle(path, self.teacher, layer=0,
                                       projection="self_attn.q_proj")
            bad=self.groups[2].clone()
            bad[0,0] = float("nan")
            fixture_bundle(path, self.teacher, self.groups[0],self.groups[1],bad)
            with self.assertRaisesRegex(ActivationDataError, "finite"):
                load_activation_bundle(path, self.teacher, layer=0,
                                       projection="self_attn.q_proj")

    def test_allows_shared_prefix_rows_between_disjoint_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"shared_prefix.safetensors"
            train,select,audit = [x.clone() for x in self.groups]
            select[0]=train[0]  # identical BOS hidden input across different prompts
            fixture_bundle(path,self.teacher,train,select,audit)
            bundle=load_activation_bundle(path,self.teacher,layer=0,
                                          projection="self_attn.q_proj")
            self.assertEqual(bundle.selection.shape[0],20)

    def test_rejects_no_provenance_or_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"capture.safetensors"
            with self.assertRaises(ActivationDataError):
                load_activation_bundle(path, self.teacher, layer=0,
                                       projection="self_attn.q_proj")
            fixture_bundle(path,self.teacher,*self.groups,
                           capture_method="random_gaussian")
            with self.assertRaisesRegex(ActivationDataError, "hook|provenance|capture"):
                load_activation_bundle(path, self.teacher, layer=0,
                                       projection="self_attn.q_proj")


class ActivationOptimizerTests(unittest.TestCase):
    def test_true_output_optimization_accepts_only_with_selection_improvement(self):
        model=fixture(residual=False)
        baseline=model.dense_weight().clone()
        teacher=baseline*1.8
        g=torch.Generator().manual_seed(7)
        cal=torch.randn(96,12,generator=g)
        sel=torch.randn(32,12,generator=g)
        audit=torch.randn(32,12,generator=g)
        rank=model.U.shape[1]
        bpw=estimated_effective_bits(8,12,rank,False)
        result=calibrate_activation_scales(model,teacher,cal,sel,audit)
        self.assertTrue(result["accepted"])
        self.assertLess(result["selection_nrmse_after"], result["selection_nrmse_before"])
        self.assertLess(result["audit_nrmse_after"], result["audit_nrmse_before"])
        self.assertEqual(result["estimated_effective_bpw"],bpw)
        self.assertTrue(result["same_bpw_verified"])
        torch.testing.assert_close(model.dense_weight(), teacher, atol=3e-5,rtol=3e-5)

    def test_selection_failure_falls_back_exactly(self):
        model=fixture(residual=False)
        baseline=model.dense_weight().clone()
        teacher=baseline.clone()
        teacher[:,0]*=2.0
        cal=torch.zeros(32,12);cal[:,0]=1.0
        sel=torch.zeros(32,12);sel[:,1]=1.0
        audit=torch.zeros(32,12);audit[:,2]=1.0
        original={n:p.detach().clone() for n,p in model.named_parameters()}
        result=calibrate_activation_scales(model,teacher,cal,sel,audit)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"],"FALLBACK_LEGACY")
        for n,p in model.named_parameters():
            torch.testing.assert_close(p,original[n],rtol=0,atol=0)
        self.assertEqual(result["selection_nrmse_after"],result["selection_nrmse_before"])

    def test_audit_cannot_influence_selection_or_candidate_weights(self):
        a=fixture(residual=True)
        b=fixture(residual=True)
        teacher=a.dense_weight()*1.3
        gen=torch.Generator().manual_seed(31)
        cal=torch.randn(48,12,generator=gen)
        sel=torch.randn(48,12,generator=gen)
        audit1=torch.randn(48,12,generator=gen)
        audit2=torch.randn(48,12,generator=gen)*2.0
        first=calibrate_activation_scales(a,teacher,cal,sel,audit1)
        second=calibrate_activation_scales(b,teacher,cal,sel,audit2)
        self.assertEqual(first["accepted"], second["accepted"])
        self.assertAlmostEqual(first["selection_nrmse_after"],
                               second["selection_nrmse_after"],places=7)
        for key in ("u1","u1_R"):
            torch.testing.assert_close(getattr(a,key),getattr(b,key),rtol=0,atol=0)

    def test_residual_paths_and_untouched_signs(self):
        model=fixture(residual=True)
        teacher=model.dense_weight()*1.5
        g=torch.Generator().manual_seed(17)
        xs=[torch.randn(48,12,generator=g) for _ in range(3)]
        old={n:p.clone() for n,p in model.named_parameters()}
        result=calibrate_activation_scales(model,teacher,*xs)
        self.assertTrue(result["accepted"])
        for name in ["U","V","U_R","V_R","u2","v1","v2","u2_R","v1_R","v2_R"]:
            torch.testing.assert_close(getattr(model,name),old[name],rtol=0,atol=0)
        self.assertTrue(result["audit_independent_of_selection"])

    def test_bad_gate_and_dimension_rejected(self):
        model=fixture(residual=False)
        teacher=model.dense_weight()
        x=torch.ones(20,12)
        with self.assertRaises(ValueError):
            calibrate_activation_scales(model, teacher, x, x[:,:-1], x)
        for gate in (-1,1,True,float("nan")):
            with self.assertRaises(ValueError):
                calibrate_activation_scales(model, teacher, x, x, x,
                                            min_relative_improvement=gate)


if __name__ == "__main__":
    unittest.main()
