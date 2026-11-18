# SPDX-License-Identifier: AGPL-3.0-only
"""install/serve.py: the container's Atlas and harness command lines from checkpoints.json, with the photo
library settings (the pack store in device memory and its NVMe tier)."""

import argparse
import shlex
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "install"))

import serve  # noqa: E402


def args(**kw):
    base = dict(checkpoint="d3-ultra-nvfp4", model="/model", port=8015, precision=None, packs=None,
                knowledge_dir=None, store_gb=0.0, store_disk_gb=0.0, store_dir="/store")
    return argparse.Namespace(**{**base, **kw})


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


class Commands(unittest.TestCase):
    def test_no_store_leaves_the_serve_block_as_it_is(self):
        env, atlas, _ = serve.commands(args())
        self.assertFalse(any(k.startswith("ATLAS_SCORE_STORE") for k in env))
        self.assertEqual(flag(atlas, "--gpu-memory-utilization"), "0.80")

    def test_a_store_sets_its_budget_and_leaves_it_room(self):
        env, atlas, _ = serve.commands(args(store_gb=20.0))
        self.assertEqual(env["ATLAS_SCORE_STORE_GB"], "20")
        self.assertNotIn("ATLAS_SCORE_STORE_DIR", env)
        # The store is held outside Atlas's memory budget: the budget makes room for it.
        self.assertEqual(float(flag(atlas, "--gpu-memory-utilization")), serve.STORE_UTIL)
        self.assertEqual(atlas.count("--gpu-memory-utilization"), 1)

    def test_an_nvme_tier_needs_a_store(self):
        env, _, _ = serve.commands(args(store_gb=20.0, store_disk_gb=300.0))
        self.assertEqual((env["ATLAS_SCORE_STORE_DIR"], env["ATLAS_SCORE_STORE_DISK_GB"]), ("/store", "300"))
        env, _, _ = serve.commands(args(store_disk_gb=300.0))
        self.assertNotIn("ATLAS_SCORE_STORE_DIR", env)

    def test_print_shows_one_command_line(self):
        _, atlas, harness = serve.commands(args(packs=24, store_gb=20.0))
        self.assertEqual(flag(atlas, "--score-packs"), "24")
        self.assertIn("tools.sparkscore.server", shlex.join(harness))


if __name__ == "__main__":
    unittest.main()
