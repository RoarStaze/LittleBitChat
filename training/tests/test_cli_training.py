import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from training.layer_train import load_projection, TrainingError

ROOT=Path(__file__).resolve().parents[2]
PREFIX='language_model.model.layers.0.self_attn.q_proj'


class TrainingCLITests(unittest.TestCase):
    def test_real_bfloat16_single_checkpoint_loader_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            shard=root/'model-00001.safetensors'
            reference=torch.randn(8,12).bfloat16()
            save_file({PREFIX+'.weight':reference},str(shard))
            single=load_projection(shard,layer=0,projection='self_attn.q_proj')
            torch.testing.assert_close(single,reference.float())
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{PREFIX+'.weight':shard.name}}))
            from_folder=load_projection(root,layer=0,projection='self_attn.q_proj')
            torch.testing.assert_close(from_folder,single)
            with self.assertRaises(ValueError):
                load_projection(root,layer=0,projection='self_attn.q_proj',max_teacher_mib=0)

    def test_training_cli_demo_and_export_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=subprocess.run([sys.executable,'-m','training.cli','demo','--out',tmp,'--steps','5',
                '--rank','8','--device','cpu'],cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(p.returncode,0,p.stderr)
            report=json.loads(p.stdout)
            self.assertTrue(Path(report['packed_file']).exists())
            self.assertEqual(report['status'],'LAYER_QAT_EXPORTED')
            self.assertFalse(report['full_model_trained'])
            self.assertTrue(report['packed_inference_verified'])
            self.assertTrue(report['teacher_is_synthetic'])

    def test_demo_with_enhanced_foundation_exports_packed_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run(
                [sys.executable, '-m', 'training.cli', 'demo', '--out', tmp,
                 '--steps', '3', '--rank', '5', '--device', 'cpu',
                 '--itq-iterations', '2', '--svd-oversample', '4',
                 '--svd-power-iters', '4', '--binary-refine-channels', '3',
                 '--binary-refine-passes', '2', '--refine-output-scales'],
                cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr)
            report = json.loads(p.stdout)
            self.assertTrue(report['packed_inference_verified'])
            self.assertEqual(report['svd_oversample'], 4)
            self.assertEqual(report['svd_power_iters'], 4)
            self.assertEqual(report['binary_refine_channels'], 3)
            self.assertEqual(report['binary_refine_passes'], 2)
            self.assertFalse(report['validated_language_quality'])

    def test_train_command_reaches_training_cli_through_root(self):
        root = ROOT / 'littlebitchat.py'
        if not root.is_file():
            self.skipTest('Run root CLI gate in full standalone repo')
        result = subprocess.run([sys.executable, str(root), 'train', '--help'], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('layer', result.stdout)
        self.assertIn('demo', result.stdout)

    def test_real_checkpoint_projection_trains_and_exports(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            teacher=root/'teacher.safetensors'
            save_file({PREFIX+'.weight':torch.randn(16,24).bfloat16()},str(teacher))
            export=root/'trained'
            p=subprocess.run([sys.executable,'-m','training.cli','layer',
                '--checkpoint',str(teacher),'--layer','0','--projection','self_attn.q_proj',
                '--rank','8','--steps','5','--device','cpu','--out',str(export)],
                cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(p.returncode,0,p.stderr)
            report=json.loads(p.stdout)
            self.assertTrue(report['teacher_weights_loaded_from_real_checkpoint'])
            self.assertFalse(report['teacher_is_synthetic'])
            self.assertTrue(report['packed_inference_verified'])
            self.assertTrue(Path(report['packed_file']).exists())

    def test_without_teacher_checkpoint_fails_closed(self):
        p=subprocess.run([sys.executable,'-m','training.cli','layer','--checkpoint','nonexistent',
          '--layer','0','--projection','mlp.up_proj','--out','ignored'],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('checkpoint',p.stderr.lower())

if __name__=='__main__': unittest.main()
