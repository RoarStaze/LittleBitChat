"""CLI cannot produce synthetic replies when trained weights are absent."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
CLI=ROOT/"littlebitchat.py"

class CommandLineContracts(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,str(CLI),*args],cwd=ROOT,capture_output=True,text=True,timeout=30)

    def test_status_reports_missing_real_weights(self):
        r=self.run_cli("status","--json")
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('"status": "MISSING_CHECKPOINT"',r.stdout)
        self.assertIn('"can_chat": false',r.stdout)

    def test_chat_refuses_without_checkpoint_even_if_unverified_flag(self):
        r=self.run_cli("chat","--allow-unverified-checkpoint")
        self.assertNotEqual(r.returncode,0)
        self.assertIn("checkpoint",r.stderr.lower())
        self.assertNotIn("LittleBit:",r.stdout)

    def test_help_includes_status_chat_and_preflight(self):
        r=self.run_cli("--help")
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn("chat",r.stdout)
        self.assertIn("status",r.stdout)
        self.assertIn("preflight",r.stdout)
if __name__=="__main__":unittest.main()
