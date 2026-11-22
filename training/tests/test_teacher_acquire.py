"""Offline-safe, license-respecting single-shard pretrained Gemma acquisition."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from training.teacher_acquire import (
    TeacherAcquireError, plan_from_index, acquire_teacher_shard, OFFICIAL_GEMMA_REPO,
)

KEY = "language_model.model.layers.0.self_attn.q_proj.weight"
SHARD = "model-00001-of-00012.safetensors"


class TeacherAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "gemma-original"
        self.calls = []
        self.info = SimpleNamespace(sha="f" * 40, siblings=[
            SimpleNamespace(rfilename="model.safetensors.index.json", size=1100),
            SimpleNamespace(rfilename=SHARD, size=20000),
            SimpleNamespace(rfilename="model-00002-of-00012.safetensors", size=20000),
        ])

    def download_file(self, *, repo_id, filename, revision, local_dir):
        self.calls.append((repo_id, filename, revision))
        self.assertEqual(repo_id, OFFICIAL_GEMMA_REPO)
        self.assertEqual(revision, self.info.sha)
        self.assertEqual(str(local_dir), str(self.out))
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if filename == "model.safetensors.index.json":
            destination.write_text(json.dumps({"weight_map": {
                KEY: SHARD,
                "language_model.model.layers.0.self_attn.k_proj.weight": SHARD,
                "language_model.model.layers.1.self_attn.q_proj.weight": "model-00002-of-00012.safetensors",
            }}))
        else:
            destination.write_bytes(b"fake-test-shard")
        return str(destination)

    def model_info(self, *, repo_id, revision, files_metadata):
        self.assertEqual(repo_id, OFFICIAL_GEMMA_REPO)
        self.assertEqual(revision, "main")
        self.assertTrue(files_metadata)
        return self.info

    def test_index_maps_real_q_projection_to_one_shard(self):
        metadata = {"weight_map": {KEY: SHARD}}
        item = plan_from_index(metadata, 0, "self_attn.q_proj")
        self.assertEqual(item["tensor_key"], KEY)
        self.assertEqual(item["shard_filename"], SHARD)
        self.assertFalse(item["complete_model_download"])

    def test_rejects_unknown_layer_projection_or_missing_key(self):
        valid = {"weight_map": {KEY: SHARD}}
        with self.assertRaises(ValueError):
            plan_from_index(valid, 62, "self_attn.q_proj")
        with self.assertRaises(ValueError):
            plan_from_index(valid, 0, "not_a_projection")
        with self.assertRaisesRegex(TeacherAcquireError, "missing|Missing"):
            plan_from_index(valid, 0, "self_attn.k_proj")

    def test_rejects_manifest_traversal_absolute_or_non_safetensors(self):
        for value in ("../escape.safetensors", "nested/model.safetensors",
                      "C:\\outside.safetensors", "model.pt", "/tmp/out.safetensors"):
            with self.subTest(value=value), self.assertRaises(TeacherAcquireError):
                plan_from_index({"weight_map": {KEY: value}}, 0, "self_attn.q_proj")

    def test_plan_does_not_download_massive_shard(self):
        report = acquire_teacher_shard(
            out=self.out, layer=0, projection="self_attn.q_proj",
            download=False, model_info=self.model_info,
            hf_download=self.download_file,
        )
        self.assertEqual(report["status"], "TEACHER_SHARD_PLAN_ONLY")
        self.assertEqual([c[1] for c in self.calls], ["model.safetensors.index.json"])
        self.assertEqual(report["shard_bytes"], 20000)
        self.assertFalse(report["shard_downloaded"])
        self.assertFalse(report["whole_model_downloaded"])
        self.assertEqual(report["pinned_revision"], "f"*40)
        self.assertEqual(report["tensor_key"], KEY)

    def test_explicit_download_fetches_only_required_shard(self):
        report = acquire_teacher_shard(
            out=self.out, layer=0, projection="self_attn.q_proj",
            download=True, model_info=self.model_info,
            hf_download=self.download_file,
        )
        self.assertEqual(report["status"], "TEACHER_SHARD_DOWNLOADED")
        self.assertEqual([c[1] for c in self.calls],
                         ["model.safetensors.index.json", SHARD])
        self.assertTrue(report["shard_downloaded"])
        self.assertTrue(Path(report["shard_path"]).is_file())
        self.assertFalse(report["source_is_pretrained_verified"])
        self.assertFalse(report["littlebit_model_trained"])

    def test_missing_size_rejected_for_download(self):
        self.info.siblings[1].size = None
        with self.assertRaisesRegex(TeacherAcquireError, "size"):
            acquire_teacher_shard(
                out=self.out, layer=0, projection="self_attn.q_proj",
                download=True, model_info=self.model_info,
                hf_download=self.download_file,
            )
        self.assertEqual([c[1] for c in self.calls], ["model.safetensors.index.json"])

    def test_download_rejects_file_larger_than_configured_cap(self):
        self.info.siblings[1].size = 10 * 1024**3
        with self.assertRaisesRegex(TeacherAcquireError, "limit|cap"):
            acquire_teacher_shard(
                out=self.out, layer=0, projection="self_attn.q_proj",
                download=True, max_shard_gib=8, model_info=self.model_info,
                hf_download=self.download_file,
            )
        self.assertEqual([c[1] for c in self.calls], ["model.safetensors.index.json"])

    def test_missing_license_or_auth_does_not_download_weight_shards(self):
        def denied(**kwargs):
            raise PermissionError("gated model")
        with self.assertRaisesRegex(TeacherAcquireError, "accept|license|login"):
            acquire_teacher_shard(out=self.out, layer=0, projection="self_attn.q_proj",
                download=True, model_info=denied, hf_download=self.download_file)
        self.assertEqual(self.calls, [])

    def test_download_path_symlink_rejected(self):
        outer = Path(self.temp.name) / "outer"
        outer.mkdir()
        self.out.symlink_to(outer, target_is_directory=True)
        with self.assertRaises(TeacherAcquireError):
            acquire_teacher_shard(
                out=self.out, layer=0, projection="self_attn.q_proj",
                download=True, model_info=self.model_info,
                hf_download=self.download_file,
            )
        self.assertEqual(self.calls, [])

    def test_cli_flags_are_exposed_through_root(self):
        import subprocess, sys
        root = Path(__file__).resolve().parents[2]
        cmd = [sys.executable, str(root / "littlebitchat.py"),
               "train", "acquire", "--help"]
        res = subprocess.run(cmd, cwd=root, text=True,
                             capture_output=True, timeout=20)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("--download", res.stdout)
        self.assertIn("--layer", res.stdout)
        self.assertIn("--projection", res.stdout)


if __name__ == "__main__":
    unittest.main()
