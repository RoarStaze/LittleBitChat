"""Exercise a genuine PyTorch forward-pre-hook capture without loading 27B."""
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from training.activation_capture import capture_projection_inputs
from training.activation_calibration import load_activation_bundle


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=nn.Module()
        self.self_attn.q_proj=nn.Linear(12,8,bias=False)

    def forward(self,x):
        return self.self_attn.q_proj(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model=nn.Module()
        self.language_model.model=nn.Module()
        self.language_model.model.layers=nn.ModuleList([TinyLayer()])

    def forward(self,x):
        return self.language_model.model.layers[0](x)


class HookCaptureTests(unittest.TestCase):
    def test_real_hook_capture_matches_linear_inputs_and_teacher(self):
        model=TinyModel().eval()
        layer=model.language_model.model.layers[0].self_attn.q_proj
        teacher=layer.weight.detach().float().clone()
        g=torch.Generator().manual_seed(48)
        batches=[[torch.randn(2,11,12,generator=g)] for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"activations.safetensors"
            result=capture_projection_inputs(
                model,teacher,calibration_batches=batches[0],
                selection_batches=batches[1],audit_batches=batches[2],
                out=out,layer=0,projection="self_attn.q_proj",
                source_model="in-process-toy-teacher",
                calibration_text_sha256="1"*64,
                selection_text_sha256="2"*64,
                audit_text_sha256="3"*64,
                max_rows=64)
            self.assertEqual(result["calibration_rows"],22)
            self.assertTrue(out.is_file())
            loaded=load_activation_bundle(out,teacher,layer=0,
                                          projection="self_attn.q_proj")
            torch.testing.assert_close(loaded.calibration,batches[0][0].reshape(-1,12))
            self.assertFalse(loaded.provenance_independently_verified)

    def test_original_model_capture_cli_requires_memory_ack(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"not_created.safetensors"
            args=[sys.executable,str(root/"littlebitchat.py"),"train","capture-activations",
                  "--checkpoint",tmp,
                  "--layer","0","--projection","self_attn.q_proj",
                  "--calibration-texts",str(Path(tmp)/"train.txt"),
                  "--selection-texts",str(Path(tmp)/"select.txt"),
                  "--audit-texts",str(Path(tmp)/"audit.txt"),
                  "--out",str(target)]
            result=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=25)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("confirm",result.stderr.lower())
            self.assertFalse(target.exists())

    def test_capture_refuses_mismatched_teacher_and_writes_nothing(self):
        model=TinyModel().eval()
        fake=torch.randn(8,12)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"bad.safetensors"
            with self.assertRaisesRegex(ValueError,"teacher|module|weight"):
                capture_projection_inputs(
                    model,fake,
                    calibration_batches=[torch.randn(1,20,12)],
                    selection_batches=[torch.randn(1,20,12)],
                    audit_batches=[torch.randn(1,20,12)],
                    out=path,layer=0,projection="self_attn.q_proj",
                    source_model="toy",calibration_text_sha256="1"*64,
                    selection_text_sha256="2"*64,
                    audit_text_sha256="3"*64)
            self.assertFalse(path.exists())


if __name__=="__main__":
    unittest.main()
