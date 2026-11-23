"""Large-scale token-type capture, frozen audit integrity and paired uncertainty tests."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

from training.tests.test_layer0_capture import Layer0CaptureTests, EMBED, NORM, Q
from training.layer_train import _teacher_sha
from training.activation_calibration import load_activation_bundle
from training.frozen_audit import (
    FrozenAuditError, capture_vocabulary_checkpoint,
    verify_frozen_audit_lock, paired_audit_bootstrap
)

ROOT=Path(__file__).resolve().parents[2]


class FakeBackend:
    def id_to_token(self, tid):
        if not 0 <= tid < 256:
            return None
        return ["<bos>", "<eos>", "<pad>", "<start_of_turn>",
                "<end_of_turn>", "<unused0>"][tid] if tid < 6 else f"word_{tid}"


class FakeTokenizer:
    vocab_size=256
    backend=FakeBackend()

    def _encode(self, text):
        # Allow a local text corpus to define an independent input pool.
        return [6+(sum(map(ord,word))%250) for word in text.split()]


class FrozenAuditTests(unittest.TestCase):
    def setUp(self):
        Layer0CaptureTests.setUp(self)
        self.embedding=torch.randn(256,12,generator=torch.Generator().manual_seed(56)).bfloat16()
        save_file({EMBED:self.embedding},str(self.ckpt/"embed.safetensors"))
        self.tokenizer=FakeTokenizer()
        self.lock=self.root/"frozen_audit.lock.json"

    def capture(self,out,*,seed=42,source=None):
        return capture_vocabulary_checkpoint(
            self.ckpt,out=out,audit_lock=self.lock,tokenizer=self.tokenizer,
            projection="self_attn.q_proj",layer=0,
            calibration_tokens=48,selection_tokens=24,audit_tokens=24,
            sample_seed=seed,audit_seed=20261122,
            source_text_file=source,allow_test_shapes=True)

    def test_creates_large_disjoint_capture_and_immutable_audit_lock(self):
        out=self.root/"first.safetensors"
        report=self.capture(out)
        self.assertEqual(report["status"],"FROZEN_TOKEN_AUDIT_CAPTURE")
        self.assertEqual(report["unique_token_counts"],
                         {"calibration":48,"selection":24,"audit":24})
        self.assertTrue(report["frozen_audit_verified"])
        self.assertTrue(self.lock.is_file())
        bundle=load_activation_bundle(out,self.teacher.float(),layer=0,
                                      projection="self_attn.q_proj",
                                      require_token_disjoint=True)
        self.assertEqual(bundle.metadata["split_method"],"disjoint_token_ids_v1")
        for a,b in (("calibration","selection"),("calibration","audit"),
                    ("selection","audit")):
            self.assertFalse(set(bundle.token_ids[a].tolist()) &
                             set(bundle.token_ids[b].tolist()))
        self.assertTrue(set(bundle.token_ids["audit"].tolist()).isdisjoint(set(range(6))))
        audit=verify_frozen_audit_lock(bundle,self.lock,self.teacher.float(),layer=0,
                                      projection="self_attn.q_proj")
        self.assertTrue(audit["frozen_audit_verified"])
        self.assertEqual(audit["frozen_audit_rows"],24)
        self.assertEqual(audit["frozen_audit_lock_sha256"],report["frozen_audit_lock_sha256"])

    def test_locked_audit_is_identical_when_training_seed_changes(self):
        one=self.root/"one.safetensors"
        two=self.root/"two.safetensors"
        first=self.capture(one,seed=42)
        snapshot=self.lock.read_bytes()
        second=self.capture(two,seed=1043)
        self.assertEqual(snapshot,self.lock.read_bytes())
        a=load_file(str(one));b=load_file(str(two))
        torch.testing.assert_close(a["token_ids_audit"],b["token_ids_audit"],rtol=0,atol=0)
        torch.testing.assert_close(a["audit"],b["audit"],rtol=0,atol=0)
        self.assertNotEqual(a["token_ids_calibration"].tolist(),
                            b["token_ids_calibration"].tolist())
        self.assertEqual(first["frozen_audit_lock_sha256"],
                         second["frozen_audit_lock_sha256"])

    def test_frozen_lock_rejects_changed_rows_ids_and_lock_bytes(self):
        first=self.root/"first.safetensors"
        self.capture(first)
        capture=load_file(str(first))
        with safe_open(str(first),framework="pt",device="cpu") as reader:
            meta=dict(reader.metadata())
        capture["audit"][0,0]+=0.25
        tampered=self.root/"bad.safetensors"
        save_file(capture,str(tampered),metadata=meta)
        bundle=load_activation_bundle(tampered,self.teacher.float(),layer=0,
                                      projection="self_attn.q_proj",
                                      require_token_disjoint=True)
        with self.assertRaisesRegex(FrozenAuditError,"frozen|mismatch|audit"):
            verify_frozen_audit_lock(bundle,self.lock,self.teacher.float(),layer=0,
                                     projection="self_attn.q_proj")
        capture["audit"]=load_file(str(first))["audit"].clone()
        capture["token_ids_audit"][:2]=capture["token_ids_audit"][:2].flip(0)
        save_file(capture,str(tampered),metadata=meta)
        bundle=load_activation_bundle(tampered,self.teacher.float(),layer=0,
                                      projection="self_attn.q_proj",
                                      require_token_disjoint=True)
        with self.assertRaises(FrozenAuditError):
            verify_frozen_audit_lock(bundle,self.lock,self.teacher.float(),layer=0,
                                     projection="self_attn.q_proj")
        self.lock.write_text(self.lock.read_text()+" ")
        original=load_activation_bundle(first,self.teacher.float(),layer=0,
                                      projection="self_attn.q_proj",
                                      require_token_disjoint=True)
        with self.assertRaisesRegex(FrozenAuditError,"lock|canonical|mismatch"):
            verify_frozen_audit_lock(original,self.lock,self.teacher.float(),layer=0,
                                     projection="self_attn.q_proj")

    def test_lock_is_not_replaced_and_teacher_mismatch_is_fail_closed(self):
        first=self.root/"first.safetensors"
        self.capture(first)
        with self.assertRaises(FrozenAuditError):
            self.capture(first)
        contents=self.lock.read_bytes()
        with self.assertRaises(FrozenAuditError):
            capture_vocabulary_checkpoint(
                self.ckpt,out=self.root/"changed.safetensors",
                audit_lock=self.lock,tokenizer=self.tokenizer,
                calibration_tokens=48,selection_tokens=24,
                audit_tokens=25,sample_seed=42,audit_seed=20261122,
                allow_test_shapes=True)
        self.assertEqual(contents,self.lock.read_bytes())
        modified=self.teacher.float().clone()
        modified[0,0]+=1
        bundle=load_activation_bundle(first,self.teacher.float(),layer=0,
                                      projection="self_attn.q_proj",
                                      require_token_disjoint=True)
        with self.assertRaises(FrozenAuditError):
            verify_frozen_audit_lock(bundle,self.lock,modified,
                                     layer=0,projection="self_attn.q_proj")

    def test_external_fingerprint_rejects_replaced_lock(self):
        path=self.root/"frozen_capture.safetensors"
        report=self.capture(path)
        bundle=load_activation_bundle(
            path,self.teacher.float(),layer=0,projection="self_attn.q_proj",
            require_token_disjoint=True)
        good=report["frozen_audit_lock_sha256"]
        matched=verify_frozen_audit_lock(
            bundle,self.lock,self.teacher.float(),layer=0,
            projection="self_attn.q_proj",expected_sha256=good)
        self.assertEqual(matched["frozen_audit_lock_sha256"],good)
        with self.assertRaisesRegex(FrozenAuditError,"expected|fingerprint"):
            verify_frozen_audit_lock(
                bundle,self.lock,self.teacher.float(),layer=0,
                projection="self_attn.q_proj",expected_sha256="0"*64)
        with self.assertRaisesRegex(FrozenAuditError,"expected|fingerprint"):
            capture_vocabulary_checkpoint(
                self.ckpt,out=self.root/"denied_again.safetensors",
                audit_lock=self.lock,tokenizer=self.tokenizer,
                calibration_tokens=48,selection_tokens=24,audit_tokens=24,
                allow_test_shapes=True,expected_audit_lock_sha256="0"*64)
        self.assertFalse((self.root/"denied_again.safetensors").exists())

    def test_restricts_special_token_ids_and_small_pools(self):
        with self.assertRaisesRegex(FrozenAuditError,"insufficient|unique|eligible"):
            capture_vocabulary_checkpoint(
                self.ckpt,out=self.root/"small.safetensors",
                audit_lock=self.lock,tokenizer=self.tokenizer,
                calibration_tokens=240,selection_tokens=20,audit_tokens=20,
                allow_test_shapes=True)
        self.assertFalse(self.lock.exists())
        self.assertFalse((self.root/"small.safetensors").exists())

    def test_optional_local_text_file_is_not_mistaken_for_uniform_vocabulary(self):
        corpus=self.root/"corpus.txt"
        corpus.write_text(" ".join(f"new_word_{i}_{i*i}" for i in range(500)))
        with self.assertRaisesRegex(FrozenAuditError,"insufficient|unique|eligible"):
            capture_vocabulary_checkpoint(
                self.ckpt,out=self.root/"small_text.safetensors",
                audit_lock=self.lock,tokenizer=self.tokenizer,
                calibration_tokens=200,selection_tokens=100,audit_tokens=100,
                source_text_file=corpus,allow_test_shapes=True)
        self.assertFalse(self.lock.exists())

    def test_cli_audit_and_calibration_accept_lock_and_refuse_mutation(self):
        output=self.root/"source.safetensors"
        self.capture(output)
        base=[sys.executable,str(ROOT/"littlebitchat.py"),"train"]
        common=["--checkpoint",str(self.ckpt),"--activation-data",str(output),
                "--layer","0","--projection","self_attn.q_proj",
                "--frozen-audit-lock",str(self.lock),
                "--require-token-disjoint"]
        audit=subprocess.run(base+["activation-audit"]+common,cwd=ROOT,
                             capture_output=True,text=True,timeout=60)
        self.assertEqual(audit.returncode,0,audit.stderr)
        a=json.loads(audit.stdout)
        self.assertTrue(a["frozen_audit_verified"])
        self.assertTrue(a["source_reconstruction_verified"])
        dest=self.root/"output"
        job=subprocess.run(base+["activation-calibrate"]+common+
            ["--rank","3","--no-residual","--max-bpw","16",
             "--itq-iterations","2","--out",str(dest)],cwd=ROOT,
            capture_output=True,text=True,timeout=120)
        self.assertEqual(job.returncode,0,job.stderr)
        report=json.loads(job.stdout)
        self.assertTrue(report["frozen_audit_verified"])
        self.assertEqual(report["frozen_audit_lock_sha256"],a["frozen_audit_lock_sha256"])
        self.assertTrue(report["same_bpw_verified"])
        self.assertEqual(report["audit_bootstrap_resamples"],1000)
        self.assertIn("audit_relative_improvement_ci95_pct",report)
        self.lock.write_text(self.lock.read_text()+" ")
        denied=subprocess.run(base+["activation-calibrate"]+common+
            ["--rank","3","--no-residual","--max-bpw","16",
             "--itq-iterations","2","--out",str(self.root/"must_not_exist")],
            cwd=ROOT,capture_output=True,text=True,timeout=45)
        self.assertNotEqual(denied.returncode,0)
        self.assertFalse((self.root/"must_not_exist").exists())

    def test_paired_bootstrap_confidence_is_positive_for_uniform_improvement(self):
        g=torch.Generator().manual_seed(123)
        before=torch.randn(128,6,generator=g)
        after=before*0.75
        result=paired_audit_bootstrap(before,after,resamples=1000,seed=7)
        self.assertAlmostEqual(result["audit_relative_nrmse_improvement_pct"],25,places=4)
        ci=result["audit_relative_improvement_ci95_pct"]
        self.assertAlmostEqual(ci[0],25,places=4)
        self.assertAlmostEqual(ci[1],25,places=4)


if __name__=="__main__":
    unittest.main()
