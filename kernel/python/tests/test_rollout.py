from __future__ import annotations

import filecmp
import tempfile
import unittest
from pathlib import Path

from mtg_kernel_rl.rollout import run_episodes

from fixtures import fake_launcher


class RolloutTest(unittest.TestCase):
    def test_fake_rollout_artifacts_are_byte_deterministic_and_terminal_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            launcher = fake_launcher(tmp, "valid")
            out_a = tmp / "a"
            out_b = tmp / "b"
            run_episodes(env_bin=launcher, out_dir=out_a, episodes=2, base_seed=71501, max_decisions=8, p0="uniform", p1="uniform")
            run_episodes(env_bin=launcher, out_dir=out_b, episodes=2, base_seed=71501, max_decisions=8, p0="uniform", p1="uniform")
            self.assertTrue(filecmp.cmp(out_a / "run.json", out_b / "run.json", shallow=False))
            self.assertTrue(filecmp.cmp(out_a / "episodes.jsonl", out_b / "episodes.jsonl", shallow=False))
            run_text = (out_a / "run.json").read_text(encoding="utf-8")
            episodes_text = (out_a / "episodes.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(str(out_a), run_text)
            self.assertNotIn("observation", run_text)
            self.assertNotIn("legal_actions", run_text)
            self.assertIn('"halted":0', run_text)
            self.assertIn('"truncated":0', run_text)
            self.assertIn('"terminal_outcome"', episodes_text)


if __name__ == "__main__":
    unittest.main()
