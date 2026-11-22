"""End-to-end activation calibration CLI must gate and export verified factors."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from training.tests.test_activation_calibration import fixture_bundle

ROOT=Path(__file__).resolve().parents[2]
KEY="language_model.model.layers.0.self_attn.q_proj.weight"


class ActivationCalibrationCLITests(unittest.TestCase):
    def test_cli_calibrates_and_exports_without_claiming_language_quality(self):
        g=torch.Generator().manual_seed(91)
        weight=torch.randn(8,12,generator=g).bfloat16()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            teacher=root/"teacher.safetensors"
            bundle=root/"capture.safetensors"
            out=root/"export"
            save_file({KEY:weight},str(teacher))
            xs=[torch.randn(32,12,generator=g) for _ in range(3)]
            fixture_bundle(bundle,weight.float(),*xs)
            p=subprocess.run(
                [sys.executable,str(ROOT/"littlebitchat.py"),"train","activation-calibrate",
                 "--checkpoint",str(teacher),"--activation-data",str(bundle),
                 "--layer","0","--projection","self_attn.q_proj",
                 "--rank","3","--no-residual","--itq-iterations","2",
                 "--max-bpw","16","--out",str(out)],
                cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(p.returncode,0,p.stderr)
            report=json.loads(p.stdout)
            self.assertIn(report["decision"],("ACTIVATION_CANDIDATE_ACCEPTED","FALLBACK_LEGACY"))
            self.assertTrue(report["same_bpw_verified"])
            self.assertTrue(report["packed_inference_verified"])
            self.assertTrue(Path(report["packed_file"]).exists())
            self.assertFalse(report["activation_source_provenance_independently_verified"])
            self.assertFalse(report["validated_language_quality"])
            self.assertTrue(report["audit_independent_of_selection"])

    def test_missing_capture_fails_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            teacher=root/"teacher.safetensors"
            output=root/"unused"
            save_file({KEY:torch.randn(8,12).bfloat16()},str(teacher))
            cmd=[sys.executable,str(ROOT/"littlebitchat.py"),"train","activation-calibrate",
                 "--checkpoint",str(teacher),"--activation-data",str(root/"missing.safetensors"),
                 "--layer","0","--projection","self_attn.q_proj",
                 "--rank","3","--no-residual","--max-bpw","16","--out",str(output)]
            p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(p.returncode,0)
            self.assertIn("Activation",p.stderr)
            self.assertFalse(output.exists())


if __name__=="__main__":
    unittest.main()
