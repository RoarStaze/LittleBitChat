"""Token-disjoint activation validation: fail closed on leaked first-layer token IDs."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from training.activation_calibration import ActivationDataError, load_activation_bundle
from training.layer0_capture import capture_layer0_from_original_weights, Layer0CaptureError
from training.tests.test_layer0_capture import Layer0CaptureTests, EMBED, NORM, Q
from training.token_disjoint import select_token_disjoint_ids, TokenDisjointError

ROOT=Path(__file__).resolve().parents[2]


class TokenizerWithSharedPrefix:
    def __init__(self, groups):
        prompts=sum(groups.values(),[])
        self.bases={s:30+10*i for i,s in enumerate(prompts)}

    def encode_conversation(self, turns):
        return [2,3,4,5,6,7,17,18] + list(range(
            self.bases[turns[0][1]],self.bases[turns[0][1]]+8))


class TokenPartitionTests(unittest.TestCase):
    def setUp(self):
        self.groups={
            "calibration":["cal zero","cal one","cal two"],
            "selection":["sel zero","sel one","sel two"],
            "audit":["aud zero","aud one","aud two"],
        }
        self.tokenizer=TokenizerWithSharedPrefix(self.groups)

    def test_partition_removes_shared_prefixes_and_has_no_repeated_token_ids(self):
        plan=select_token_disjoint_ids(
            self.tokenizer,self.groups,max_tokens=64,max_rows=16,
            min_unique_tokens=16,vocab_size=256)
        ids=plan["ids"]
        self.assertGreaterEqual(plan["shared_dropped"],8)
        self.assertEqual(plan["sampled_unique_counts"],{
            "calibration":16,"selection":16,"audit":16})
        for name in ids:
            self.assertEqual(len(ids[name]),len(set(ids[name])))
            self.assertFalse(set(ids[name]) & {2,3,4,5,6,7,17,18})
        self.assertFalse(set(ids["calibration"]) & set(ids["selection"]))
        self.assertFalse(set(ids["calibration"]) & set(ids["audit"]))
        self.assertFalse(set(ids["selection"]) & set(ids["audit"]))
        repeat=select_token_disjoint_ids(
            self.tokenizer,self.groups,max_tokens=64,max_rows=16,
            min_unique_tokens=16,vocab_size=256)
        self.assertEqual(ids,repeat["ids"])

    def test_fail_closed_if_any_split_lacks_unique_tokens(self):
        with self.assertRaisesRegex(TokenDisjointError,"unique|token-disjoint|more"):
            select_token_disjoint_ids(
                self.tokenizer,self.groups,max_tokens=64,max_rows=64,
                min_unique_tokens=64,vocab_size=256)
        with self.assertRaisesRegex(TokenDisjointError,"min_unique_tokens"):
            select_token_disjoint_ids(
                self.tokenizer,self.groups,max_tokens=64,max_rows=16,
                min_unique_tokens=15,vocab_size=256)


class CaptureAndValidationTests(unittest.TestCase):
    """Inherits original capture checks and adds token-disjoint guarantees."""
    def setUp(self):
        Layer0CaptureTests.setUp(self)
        self.groups={
            "calibration":["cal zero","cal one","cal two"],
            "selection":["sel zero","sel one","sel two"],
            "audit":["aud zero","aud one","aud two"],
        }
        self.tokenizer=TokenizerWithSharedPrefix(self.groups)
        self.embedding=torch.randn(256,12,generator=torch.Generator().manual_seed(13)).bfloat16()
        save_file({EMBED:self.embedding},str(self.ckpt/"embed.safetensors"))

    def _capture(self):
        out=self.root/"strict_capture.safetensors"
        report=capture_layer0_from_original_weights(
            self.ckpt,out=out,projection="self_attn.q_proj",
            tokenizer=self.tokenizer,prompts=self.groups,
            max_rows=16,max_tokens=64,min_unique_tokens=16,
            token_disjoint=True,allow_test_shapes=True)
        return out,report

    def test_strict_capture_reconstructs_original_embedding_rows_and_labels(self):
        path,report=self._capture()
        self.assertTrue(report["token_disjoint_verified"])
        self.assertEqual(report["calibration_rows"],16)
        self.assertEqual(report["selection_rows"],16)
        self.assertEqual(report["audit_rows"],16)
        self.assertGreater(report["shared_token_ids_dropped"],0)
        bundle=load_activation_bundle(path,self.teacher.float(),layer=0,
                                      projection="self_attn.q_proj",
                                      require_token_disjoint=True)
        self.assertTrue(bundle.token_disjoint_verified)
        self.assertEqual(bundle.unique_token_counts,dict.fromkeys(
            ("calibration","selection","audit"),16))
        self.assertEqual(bundle.metadata["split_method"],"disjoint_token_ids_v1")
        with safe_open(str(path),framework="pt",device="cpu") as opened:
            original_ids=opened.get_tensor("token_ids_calibration").tolist()
        expected=torch.stack([
            self.embedding[tid].clone() for tid in original_ids])
        from training.layer0_capture import gemma3_layer0_qkv_inputs
        reconstructed=gemma3_layer0_qkv_inputs(original_ids,expected,self.norm)
        torch.testing.assert_close(bundle.calibration,reconstructed,rtol=0,atol=0)

    def test_strict_loader_rejects_cross_split_token_reuse_even_if_rows_differ(self):
        path,_=self._capture()
        data=load_file(str(path))
        with safe_open(str(path),framework="pt",device="cpu") as opened:
            meta=dict(opened.metadata())
        data["token_ids_selection"][0]=data["token_ids_calibration"][0]
        # Make matrix row numerically different: still invalid by token identity.
        data["selection"][0]+=0.1
        tampered=self.root/"tampered.safetensors"
        save_file(data,str(tampered),metadata=meta)
        with self.assertRaisesRegex(ActivationDataError,"token|overlap"):
            load_activation_bundle(tampered,self.teacher.float(),layer=0,
                                   projection="self_attn.q_proj",
                                   require_token_disjoint=True)

    def test_strict_loader_rejects_missing_or_mismatched_id_arrays(self):
        path,_=self._capture()
        data=load_file(str(path))
        with safe_open(str(path),framework="pt",device="cpu") as opened:
            meta=dict(opened.metadata())
        del data["token_ids_audit"]
        tampered=self.root/"incomplete.safetensors"
        save_file(data,str(tampered),metadata=meta)
        with self.assertRaisesRegex(ActivationDataError,"token|split|keys"):
            load_activation_bundle(tampered,self.teacher.float(),layer=0,
                                   projection="self_attn.q_proj",require_token_disjoint=True)
        data["token_ids_audit"]=torch.arange(15,dtype=torch.int64)
        save_file(data,str(tampered),metadata=meta)
        with self.assertRaisesRegex(ActivationDataError,"row|shape|length|count"):
            load_activation_bundle(tampered,self.teacher.float(),layer=0,
                                   projection="self_attn.q_proj",require_token_disjoint=True)

    def test_existing_old_prompt_capture_remains_readable_but_not_strict(self):
        out=self.root/"old.safetensors"
        capture_layer0_from_original_weights(
            self.ckpt,out=out,projection="self_attn.q_proj",
            tokenizer=self.tokenizer,prompts=self.groups,
            max_rows=64,max_tokens=64,allow_test_shapes=True)
        old=load_activation_bundle(out,self.teacher.float(),
                                   layer=0,projection="self_attn.q_proj")
        self.assertFalse(old.token_disjoint_verified)
        with self.assertRaisesRegex(ActivationDataError,"token-disjoint"):
            load_activation_bundle(out,self.teacher.float(),layer=0,
                                   projection="self_attn.q_proj",
                                   require_token_disjoint=True)

    def test_capture_rejects_tiny_corpus_before_export(self):
        with self.assertRaisesRegex(Layer0CaptureError,"unique|token-disjoint|more"):
            capture_layer0_from_original_weights(
                self.ckpt,out=self.root/"denied.safetensors",
                projection="self_attn.q_proj",tokenizer=self.tokenizer,
                prompts=self.groups,max_rows=64,max_tokens=64,
                min_unique_tokens=64,token_disjoint=True,allow_test_shapes=True)
        self.assertFalse((self.root/"denied.safetensors").exists())

    def test_attestation_rejects_relabelled_ids_with_unchanged_rows(self):
        path,_=self._capture()
        data=load_file(str(path))
        with safe_open(str(path),framework="pt",device="cpu") as opened:
            meta=dict(opened.metadata())
        old=data["token_ids_audit"].clone()
        data["token_ids_audit"][0],data["token_ids_audit"][1]=old[1],old[0]
        changed=self.root/"relabelled.safetensors"
        save_file(data,str(changed),metadata=meta)
        # All three ID sets are still disjoint. A label-only validation
        # would incorrectly accept this altered activation bundle.
        bundle=load_activation_bundle(
            changed,self.teacher.float(),layer=0,projection="self_attn.q_proj",
            require_token_disjoint=True)
        self.assertTrue(bundle.token_disjoint_verified)
        cmd=[sys.executable,str(ROOT/"littlebitchat.py"),"train","activation-audit",
             "--checkpoint",str(self.ckpt),"--activation-data",str(changed),
             "--layer","0","--projection","self_attn.q_proj",
             "--require-token-disjoint"]
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=35)
        self.assertNotEqual(result.returncode,0)
        self.assertRegex(result.stderr,"reconstruction|match|mismatch|source")

    def test_strict_calibration_and_read_only_audit_cli(self):
        path,report=self._capture()
        audit=subprocess.run([
            sys.executable,str(ROOT/"littlebitchat.py"),"train","activation-audit",
            "--checkpoint",str(self.ckpt),"--activation-data",str(path),
            "--layer","0","--projection","self_attn.q_proj",
            "--require-token-disjoint"],cwd=ROOT,capture_output=True,text=True,timeout=35)
        self.assertEqual(audit.returncode,0,audit.stderr)
        a=json.loads(audit.stdout)
        self.assertTrue(a["token_disjoint_verified"])
        self.assertEqual(a["unique_token_counts"]["audit"],16)
        dest=self.root/"trained"
        result=subprocess.run([
            sys.executable,str(ROOT/"littlebitchat.py"),"train","activation-calibrate",
            "--checkpoint",str(self.ckpt),"--activation-data",str(path),
            "--layer","0","--projection","self_attn.q_proj",
            "--rank","3","--no-residual","--itq-iterations","2",
            "--max-bpw","16","--require-token-disjoint",
            "--out",str(dest)],cwd=ROOT,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr)
        stats=json.loads(result.stdout)
        self.assertTrue(stats["same_bpw_verified"])
        self.assertTrue(stats["token_disjoint_verified"])
        self.assertTrue(stats["packed_inference_verified"])
        self.assertEqual(stats["unique_token_counts"]["audit"],16)

if __name__=="__main__":
    unittest.main()
