"""The full-model streaming orchestration must not retain multiple packed blocks."""
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"engine"))
from chat_runtime import (run_streamed_blocks, inspect_readiness,
                          CheckpointEmbeddings, ChatUnavailable)
from gemma3_tiny_lm import make_demo_model
from gemma3_checkpoint_stream import write_synthetic_checkpoint
from safetensors_layer import _ShardGroup

class FakeBlockLoader:
    def __init__(self):self.active=0;self.peak=0;self.loaded=[]
    @contextmanager
    def cuda_block(self,index,*,pool_limit_bytes):
        self.active+=1;self.peak=max(self.peak,self.active);self.loaded.append(index)
        try:yield {"value":index},None
        finally:self.active-=1

class StreamContracts(unittest.TestCase):
    def test_62_layer_walk_retains_at_most_one_packed_block(self):
        fake=FakeBlockLoader()
        result=run_streamed_blocks(fake,62,lambda i,ops,x:x+ops["value"],0,pool_limit_bytes=1024)
        self.assertEqual(fake.peak,1)
        self.assertEqual(fake.active,0)
        self.assertEqual(fake.loaded,list(range(62)))
        self.assertEqual(result,sum(range(62)))

    def test_loader_clears_on_exception(self):
        fake=FakeBlockLoader()
        def fail(i,ops,x):
            if i==4:raise RuntimeError("out of memory")
            return x+1
        with self.assertRaises(RuntimeError):
            run_streamed_blocks(fake,62,fail,0,pool_limit_bytes=1024)
        self.assertEqual(fake.active,0)
        self.assertEqual(fake.loaded,[0,1,2,3,4])

    def test_no_checkpoint_is_not_chat_ready(self):
        r=inspect_readiness()
        self.assertEqual(r["status"],"MISSING_CHECKPOINT")
        self.assertFalse(r["can_chat"])
        self.assertFalse(r["trained_weights_verified"])

    def test_readiness_never_claims_trained_model_from_filename(self):
        with self.assertRaises(ChatUnavailable):
            inspect_readiness(checkpoint=Path("model-not-here"))

    def test_embedding_rows_read_from_disk_without_materializing_full_table(self):
        with tempfile.TemporaryDirectory() as temp:
            original=make_demo_model(backend="cpu",depth=6,seed=19)
            path=Path(temp)/"weights.safetensors"
            write_synthetic_checkpoint(path,original)
            table=CheckpointEmbeddings(_ShardGroup(path),expected_vocab=32,expected_hidden=8)
            np.testing.assert_allclose(table.rows(3,2),original.embedding[3:5],atol=0)
            np.testing.assert_allclose(table.row(9),original.embedding[9],atol=0)
            self.assertEqual(table.dtype,"F32")
            with self.assertRaises(ValueError):table.rows(31,2)
            with self.assertRaises(ValueError):table.row(-1)
if __name__=="__main__":unittest.main()
