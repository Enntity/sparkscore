# SPDX-License-Identifier: AGPL-3.0-only
"""nvfp4_convert.py on a tiny synthetic checkpoint (needs torch and safetensors).

A reference in the ModelOpt MIXED_PRECISION layout (NVFP4 MLP and lm_head, FP8
attention with input_scale, BF16 MTP head) is built with the tool's own quantizers;
then a wrapped source with an lm_head must convert to exactly those tensors, and a
bare d3-style source (no `model.` wrapper, no lm_head, no MTP, a code readout) must
convert to the reference's structure with the readout as lm_head.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("torch", "safetensors"))
TOOLS = Path(__file__).resolve().parents[1] / "tools"
H, I, V = 64, 96, 300
CODES = [7, 3, 250, 41]


@unittest.skipUnless(HAVE_DEPS, "needs torch and safetensors")
class Convert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import save_file

        sys.path.insert(0, str(TOOLS))
        import nvfp4_convert as nc

        cls.torch, cls.nc, cls.save = torch, nc, save_file
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        g = torch.Generator().manual_seed(0)
        rnd = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(torch.bfloat16)  # noqa: E731
        lp = "model.language_model.layers.0"
        cls.base = {
            "model.language_model.embed_tokens.weight": rnd(V, H),
            "model.language_model.norm.weight": rnd(H),
            f"{lp}.mlp.gate_proj.weight": rnd(I, H),
            f"{lp}.mlp.up_proj.weight": rnd(I, H) * 3,
            f"{lp}.mlp.down_proj.weight": rnd(H, I),
            f"{lp}.self_attn.q_proj.weight": rnd(H, H),
            "model.visual.patch_embed.proj.weight": rnd(8, 3),
            "lm_head.weight": rnd(V, H),
        }
        mtp = {"mtp.fc.weight": rnd(H, 2 * H)}
        layers = {f"{lp}.mlp.{p}": {"quant_algo": "W4A16_NVFP4", "group_size": 16} for p in ("gate_proj", "up_proj", "down_proj")}
        layers["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        layers[f"{lp}.self_attn.q_proj"] = {"quant_algo": "FP8"}
        # the reference: the tool's own quantizers over the base weights
        ref = {k: v for k, v in cls.base.items() if k.rpartition(".")[0] not in layers}
        ref.update(mtp)
        conv = nc.Converter.__new__(nc.Converter)
        conv.layers, conv.dev = layers, "cpu"
        conv.src = type("S", (), {"get": staticmethod(lambda n: cls.base[n])})()
        for m, q in layers.items():
            ref.update(conv._module(m, q["quant_algo"]))
        ref[f"{lp}.self_attn.q_proj.input_scale"] = torch.tensor(0.0123)
        cls.ref_tensors = ref
        cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
               "text_config": {"vocab_size": V, "hidden_size": H, "dtype": "bfloat16", "mtp_num_hidden_layers": 1}}
        qc = {"quant_method": "modelopt", "quantized_layers": dict(layers), "ignore": ["mtp*"],
              "config_groups": {"g": {"targets": sorted(layers)}}}
        cls.ref = cls.write(root / "ref", ref, dict(cfg, quantization_config=qc),
                            {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "MIXED_PRECISION",
                             "quantized_layers": dict(layers), "exclude_modules": ["mtp*"]}})
        cls.wrapped = cls.write(root / "wrapped", dict(cls.base, **mtp), cfg)
        bare = {k[len("model."):]: v for k, v in cls.base.items() if k.startswith("model.")}
        cls.readout = torch.randn(len(CODES), H, generator=g, dtype=torch.float32)
        cls.d3 = cls.write(root / "d3", bare, dict(cfg, architectures=["Qwen3_5Model"], auto_map={"AutoModel": "x.Y"}))
        save_file({"weight": cls.readout}, os.path.join(cls.d3, "readout.safetensors"))
        Path(cls.d3, "decision_config.json").write_text(json.dumps({"token_ids": CODES}))
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def write(cls, path, tensors, config, hf_quant=None):
        path.mkdir()
        shards = {"a.safetensors": {}, "b.safetensors": {}}
        for i, (k, v) in enumerate(sorted(tensors.items())):
            shards["ab"[i % 2] + ".safetensors"][k] = v.contiguous()
        wm = {}
        for f, t in shards.items():
            cls.save(t, str(path / f), metadata={"format": "pt"})
            wm.update(dict.fromkeys(t, f))
        (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}))
        (path / "config.json").write_text(json.dumps(config))
        if hf_quant:
            (path / "hf_quant_config.json").write_text(json.dumps(hf_quant))
        return str(path)

    def run_tool(self, *argv):
        sys.argv = ["nvfp4_convert.py", *argv]
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as done:
            self.nc.main()
        return done.exception.code, out.getvalue()

    def load(self, path):
        ck = self.nc.Checkpoint(path)
        return {n: ck.get(n) for n in ck.where}

    def test_e2m1_ties_to_even(self):
        x = self.torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 7.0])
        vals = self.torch.tensor(self.nc.E2M1_VALUES + [-v for v in self.nc.E2M1_VALUES])
        got = vals[self.nc.e2m1_codes(x).long()].tolist()
        self.assertEqual(got, [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 6.0])

    def test_wrapped_source_reproduces_reference(self):
        out = str(self.root / "out-wrapped")
        code, _ = self.run_tool("convert", "--src", self.wrapped, "--ref", self.ref, "--out", out, "--device", "cpu")
        self.assertEqual(code, None)
        got = self.load(out)
        self.assertEqual(sorted(got), sorted(self.ref_tensors))
        for n, t in self.ref_tensors.items():
            self.assertTrue(self.torch.equal(got[n].view(-1).view(self.torch.uint8),
                                             t.contiguous().view(-1).view(self.torch.uint8)), n)

    def test_bare_readout_source(self):
        out = str(self.root / "out-d3")
        code, log = self.run_tool("convert", "--src", self.d3, "--ref", self.ref, "--out", out,
                                  "--readout", self.d3, "--drop-mtp", "--device", "cpu")
        self.assertEqual(code, None, log)
        got = self.load(out)
        lm = got["lm_head.weight"]
        self.assertEqual((lm.dtype, tuple(lm.shape)), (self.torch.bfloat16, (V, H)))
        self.assertTrue(self.torch.equal(lm[CODES], self.readout.to(self.torch.bfloat16)))
        rest = self.torch.ones(V, dtype=self.torch.bool)
        rest[CODES] = False
        self.assertEqual(lm[rest].abs().max().item(), 0.0)
        self.assertFalse(any(n.startswith("mtp.") or n.startswith("lm_head.weight_scale") for n in got))
        for n, t in self.ref_tensors.items():  # everything else as the reference, built from the same weights
            if not n.startswith(("mtp.", "lm_head.")):
                self.assertTrue(self.torch.equal(got[n].view(-1).view(self.torch.uint8),
                                                 t.contiguous().view(-1).view(self.torch.uint8)), n)
        cfg = json.loads(Path(out, "config.json").read_text())
        self.assertEqual(cfg["architectures"], ["Qwen3_5ForConditionalGeneration"])
        self.assertNotIn("auto_map", cfg)
        self.assertEqual(cfg["text_config"]["mtp_num_hidden_layers"], 0)
        self.assertIn("lm_head", cfg["quantization_config"]["ignore"])
        self.assertNotIn("lm_head", cfg["quantization_config"]["quantized_layers"])
        hq = json.loads(Path(out, "hf_quant_config.json").read_text())["quantization"]
        self.assertIn("lm_head", hq["exclude_modules"])
        self.assertTrue(Path(out, "readout.safetensors").is_file())
        code, log = self.run_tool("validate", "--src", self.d3, "--ref", self.ref, "--out", out,
                                  "--readout", self.d3, "--device", "cpu")
        self.assertEqual(code, 0, log)
        self.assertIn("VALID", log)


if __name__ == "__main__":
    unittest.main()
