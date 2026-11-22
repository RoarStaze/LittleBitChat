"""Small, exact layer-0 Gemma 3 activation capture without a full-model load."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from training.activation_calibration import load_activation_bundle
from training.layer_train import _teacher_sha
from training.layer0_capture import (
    Layer0CaptureError, inspect_layer0_source,
    gemma3_layer0_qkv_inputs, capture_layer0_from_original_weights,
)


EMBED = "language_model.model.embed_tokens.weight"
NORM = "language_model.model.layers.0.input_layernorm.weight"
Q = "language_model.model.layers.0.self_attn.q_proj.weight"


class StubGemmaTokenizer:
    """Fixed distinct prompt token IDs with shared BOS/user-prefix."""
    def __init__(self, prompts):
        self.indices = {p: 12+i for i,p in enumerate(prompts)}

    def encode_conversation(self, turns):
        assert len(turns)==1 and turns[0][0]=="user"
        return [2, 3, 4, self.indices[turns[0][1]], 5, 6, 7]


class Layer0CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ckpt = self.root / "teacher"
        self.ckpt.mkdir()
        gen = torch.Generator().manual_seed(314)
        self.embedding=torch.randn(48, 12, generator=gen).bfloat16()
        self.norm=torch.randn(12,generator=gen).bfloat16()
        self.teacher=torch.randn(8,12,generator=gen).bfloat16()
        save_file({EMBED:self.embedding},str(self.ckpt/"embed.safetensors"))
        save_file({NORM:self.norm,Q:self.teacher},str(self.ckpt/"layer.safetensors"))
        manifest={"weight_map":{EMBED:"embed.safetensors",
                                NORM:"layer.safetensors",Q:"layer.safetensors"}}
        (self.ckpt/"model.safetensors.index.json").write_text(json.dumps(manifest))
        self.groups={
            "calibration":["cal one","cal two","cal three"],
            "selection":["sel one","sel two","sel three"],
            "audit":["audit one","audit two","audit three"]}
        self.tokenizer=StubGemmaTokenizer(sum(self.groups.values(),[]))

    def test_preflight_reports_weights_without_tokenizer_and_config(self):
        info=inspect_layer0_source(self.ckpt,projection="self_attn.q_proj")
        self.assertTrue(info["needed_weight_shards_present"])
        self.assertFalse(info["tokenizer_json_present"])
        self.assertFalse(info["config_json_present"])
        self.assertEqual(info["source_layer"],0)
        self.assertFalse(info["ready_for_capture"])
        self.assertEqual(info["missing_required_tensors"],[])
        self.assertEqual(info["embedding_shape"],[48,12])
        self.assertEqual(info["norm_shape"],[12])
        self.assertEqual(info["projection_shape"],[8,12])

    def test_exact_bf16_forward_equivalence_to_embeddings_and_gemma_rmsnorm(self):
        ids=[2, 3, 12, 7]
        expected=self.embedding[ids]*torch.tensor(12**.5,dtype=torch.bfloat16)
        exact=(expected.float()*torch.rsqrt(
            expected.float().square().mean(dim=-1,keepdim=True)+1e-6))
        exact=(exact*(1+self.norm.float())).to(expected.dtype).float()
        observed=gemma3_layer0_qkv_inputs(
            ids, self.embedding, self.norm, eps=1e-6)
        torch.testing.assert_close(observed,exact,rtol=0,atol=0)

    def test_capture_only_needed_embedding_rows_and_reuse_existing_calibration(self):
        out=self.root/"capture.safetensors"
        result=capture_layer0_from_original_weights(
            self.ckpt, out=out,projection="self_attn.q_proj",
            tokenizer=self.tokenizer,prompts=self.groups,
            max_rows=64,max_tokens=64, allow_test_shapes=True)
        self.assertEqual(result["status"],"LAYER0_EXACT_EMBED_NORM_CAPTURE")
        self.assertLess(result["unique_embedding_rows_read"],48)
        self.assertEqual(result["capture_method"],"gemma3_layer0_embedding_rmsnorm")
        self.assertFalse(result["provenance_independently_verified"])
        loaded=load_activation_bundle(out,self.teacher.float(),layer=0,
                                       projection="self_attn.q_proj")
        self.assertEqual(loaded.calibration.shape[1],12)
        self.assertTrue(torch.isfinite(loaded.audit).all())
        expected=gemma3_layer0_qkv_inputs(
            self.tokenizer.encode_conversation([("user",self.groups["calibration"][0])]),
            self.embedding,self.norm)
        torch.testing.assert_close(loaded.calibration[:7],expected,rtol=0,atol=0)
        self.assertEqual(loaded.metadata["teacher_sha256"],_teacher_sha(self.teacher.float()))

    def test_missing_embedding_shard_fails_closed(self):
        (self.ckpt/"embed.safetensors").unlink()
        info=inspect_layer0_source(self.ckpt)
        self.assertFalse(info["needed_weight_shards_present"])
        out=self.root/"capture.safetensors"
        with self.assertRaises(Layer0CaptureError):
            capture_layer0_from_original_weights(
                self.ckpt,out=out,projection="self_attn.q_proj",
                tokenizer=self.tokenizer,prompts=self.groups,
                allow_test_shapes=True)
        self.assertFalse(out.exists())

    def test_not_supported_for_deeper_layers_or_mlp(self):
        with self.assertRaisesRegex(Layer0CaptureError,"layer 0|layer-0"):
            inspect_layer0_source(self.ckpt,layer=1)
        with self.assertRaisesRegex(Layer0CaptureError,"q_proj|attention"):
            inspect_layer0_source(self.ckpt,projection="mlp.gate_proj")

    def test_preflight_rejects_manifest_path_escape(self):
        manifest=json.loads((self.ckpt/"model.safetensors.index.json").read_text())
        manifest["weight_map"][EMBED]="../outside.safetensors"
        (self.ckpt/"model.safetensors.index.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(Layer0CaptureError,"unsafe|Unsafe"):
            inspect_layer0_source(self.ckpt)

    def test_capture_validates_prompt_separation_before_writing(self):
        out=self.root/"capture.safetensors"
        prompt_sets=dict(self.groups)
        prompt_sets["audit"]=prompt_sets["selection"]
        with self.assertRaisesRegex(Layer0CaptureError,"overlap|duplicate"):
            capture_layer0_from_original_weights(
                self.ckpt,out=out,projection="self_attn.q_proj",
                tokenizer=self.tokenizer,prompts=prompt_sets,
                allow_test_shapes=True)
        self.assertFalse(out.exists())


if __name__=="__main__":
    unittest.main()
