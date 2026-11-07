# SPDX-License-Identifier: AGPL-3.0-only
"""Quantize a bf16 Qwen3.5/3.6 checkpoint (MoE or dense) into the on-disk format of a
ModelOpt MIXED_PRECISION reference (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt 0.44.0, for
the MoE models; a2genesis/Qwen3.8-27B-NVFP4, ModelOpt 0.45.0, for the dense 27B).

The reference decides everything about the format: which modules are NVFP4 W4A16
(group 16) or FP8 per-tensor, every tensor name, shard assignment, index metadata,
hf_quant_config.json and config.json's quantization_config. The source supplies the
weights. Weight quantization is ModelOpt's max calibration, reproduced bit for bit:

  NVFP4   weight_scale_2 = amax(W) / (6 * 448)                       fp32 scalar
          weight_scale   = fp8_e4m3(amax16(W) / (6 * weight_scale_2))  RTNE, 0 -> 1
          weight         = e2m1_rtne(W / (weight_scale * weight_scale_2)), two codes
                           per byte, even column in the low nibble
  FP8     weight_scale   = amax(W) / 448;  weight = fp8_e4m3(W / weight_scale)

Fused routed experts (gate_up_proj [E, 2I, H], down_proj [E, H, I]) are written per
expert; gate_proj and up_proj of one expert share weight_scale_2, because ModelOpt
calibrates one quantizer per expert over the fused gate_up_proj. The shared expert's
separate gate_proj and up_proj also share one: max(amax_gate, amax_up) / (6 * 448).
FP8 projections (q/k/v, GDN in_proj_qkv/in_proj_z) keep their own per-tensor scales.

Activation scales (`input_scale`) need calibration data, which this tool does not run:
they are copied verbatim from the reference. --keep-bf16 MODULE leaves a module the
reference quantizes (e.g. lm_head) unquantized and lists it with the excluded modules. Every other tensor is copied from the
source unchanged, dtype preserved. Per-expert MTP experts in the source are fused to
the reference's 3-D layout (a pure copy, no arithmetic).

A bare backbone export (a `Qwen3_5Model`: `language_model.*`, `visual.*`, no `model.`
wrapper) is read under the reference's `model.` names; config.json keeps the source's
values but takes the reference's `architectures` and drops `auto_map`/`custom_pipelines`
(their code is not part of the output). --readout DIR builds `lm_head.weight` for a
code-readout decision model (vllm-sr d3: `readout.safetensors` {"weight": [255, H]} and
`decision_config.json` `token_ids`): zeros except row token_ids[i] = readout[i] cast to
the backbone dtype, kept unquantized as with --keep-bf16 lm_head. --drop-mtp leaves out
the reference's MTP head when the source has none (text_config.mtp_num_hidden_layers = 0).

  convert:  nvfp4_convert.py convert --src BF16_DIR --ref REF_DIR --out OUT_DIR [--keep-bf16 lm_head]
            [--readout DIR] [--drop-mtp]
  compare:  nvfp4_convert.py compare A_DIR B_DIR      (tensor by tensor)
  validate: nvfp4_convert.py validate --src BF16_DIR --ref REF_DIR --out OUT_DIR [--readout DIR]
            (structure against the reference, sampled dequantization error against the source,
            quantizer signature of both, input_scale and lm_head checks)

Run with torch (e.g. inside nvcr.io/nvidia/pytorch). Memory stays bounded by one
output shard plus one layer's experts.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_MAX, FP8_MAX, GROUP = 6.0, 448.0, 16
E2M1_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
INDEX = "model.safetensors.index.json"
FUSED_GATE_UP = ("gate_proj", "up_proj")
# Files carried over from the source next to the weights.
AUX_FILES = (
    "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json",
    "configuration.json", "LICENSE", "NOTICE", "decision_config.json", "readout.safetensors",
)
# config.json keys naming code that is not copied to the output.
REMOTE_CODE_KEYS = ("auto_map", "custom_pipelines")
WRAPPER = "model."
EXPERT_RE = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)$")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


class Checkpoint:
    """Lazy tensor access over a sharded safetensors directory (index or not)."""

    def __init__(self, path, device="cpu"):
        self.path, self.device = path, device
        idx = os.path.join(path, INDEX)
        if os.path.exists(idx):
            self.where = load_json(idx)["weight_map"]
        else:
            self.where = {}
            for f in sorted(os.listdir(path)):
                if f.endswith(".safetensors"):
                    with safe_open(os.path.join(path, f), "pt") as h:
                        self.where.update(dict.fromkeys(h.keys(), f))
        self.handles = {}
        self.used = set()
        # A bare backbone export has no `model.` wrapper; it answers to the wrapped names.
        self.bare = bool(self.where) and not any(n.startswith(WRAPPER) for n in self.where)

    def key(self, name):
        if name in self.where:
            return name
        if self.bare and name.startswith(WRAPPER) and name[len(WRAPPER):] in self.where:
            return name[len(WRAPPER):]
        return None

    def __contains__(self, name):
        return self.key(name) is not None

    def get(self, name):
        name = self.key(name) or name
        f = self.where[name]
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.path, f), "pt", device=self.device)
        self.used.add(name)
        return self.handles[f].get_tensor(name)


# ---------------------------------------------------------------- quantization
def amax(w):
    return w.abs().amax().float()


def nvfp4_quantize(w, wsf2):
    """w: [..., N, K] bf16; wsf2: fp32 broadcastable to [..., 1, 1].
    -> (packed u8 [..., N, K/2], weight_scale fp8 [..., N, K/16])."""
    n, k = w.shape[-2:]
    blocks = w.view(*w.shape[:-1], k // GROUP, GROUP)
    per_block = blocks.abs().amax(dim=-1).float() / (FP4_MAX * wsf2)
    per_block[per_block == 0] = 1.0
    scale = per_block.to(torch.float8_e4m3fn)
    x = blocks / (scale.float() * wsf2).unsqueeze(-1)
    del blocks
    x = x.view(*w.shape)
    codes = e2m1_codes(x)
    del x
    packed = (codes[..., 1::2] << 4) | codes[..., 0::2]
    return packed, scale


def e2m1_codes(x):
    """fp32 -> E2M1 codes (sign << 3 | magnitude index), round to nearest, ties to even."""
    bounds = torch.tensor(E2M1_BOUNDS, device=x.device, dtype=x.dtype)
    sign = (x < 0).to(torch.uint8)
    a = x.abs()
    code = torch.searchsorted(bounds, a, out_int32=True).to(torch.uint8)
    code += ((a == bounds[1]) | (a == bounds[3]) | (a == bounds[5])).to(torch.uint8)
    return code + (sign << 3)


def fp8_quantize(w):
    scale = amax(w) / FP8_MAX
    return (w / scale).to(torch.float8_e4m3fn), scale


def nvfp4_dequantize(packed, scale, wsf2):
    lut = torch.tensor(E2M1_VALUES + [-v for v in E2M1_VALUES], device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()
    vals = lut[codes]
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // GROUP, GROUP) * scale.float().unsqueeze(-1) * wsf2
    return vals.view(*vals.shape[:-2], k)


# ---------------------------------------------------------------- conversion
def readout_lm_head(readout_dir, vocab, dtype):
    """lm_head [vocab, H] of a code-readout model: zeros except row token_ids[i] = readout[i].
    -> (weight, max abs cast error, max rel cast error over nonzero entries)."""
    ids = load_json(os.path.join(readout_dir, "decision_config.json"))["token_ids"]
    with safe_open(os.path.join(readout_dir, "readout.safetensors"), "pt") as h:
        rows = h.get_tensor("weight")
    if rows.shape[0] != len(ids) or len(set(ids)) != len(ids) or not all(0 <= i < vocab for i in ids):
        raise SystemExit(f"readout {tuple(rows.shape)} does not fit token_ids ({len(ids)}, distinct, < {vocab})")
    cast = rows.to(dtype)
    err = (cast.float() - rows.float()).abs()
    nz = rows != 0
    rel = (err[nz] / rows.float().abs()[nz]).max().item() if nz.any() else 0.0
    weight = torch.zeros(vocab, rows.shape[1], dtype=dtype)
    weight[torch.tensor(ids)] = cast
    return weight, err.max().item(), rel


class Converter:
    def __init__(self, src, ref, device, keep_bf16=(), synth=None):
        self.src = Checkpoint(src, device)
        self.ref = Checkpoint(ref)  # input_scale tensors only
        self.dev = device
        quant = load_json(os.path.join(ref, "hf_quant_config.json"))["quantization"]
        self.layers = {m: q for m, q in quant["quantized_layers"].items() if m not in keep_bf16}
        self.cache = {}
        self.synth = dict(synth or {})  # tensors built outside the source (lm_head from --readout)
        for name in self.synth:
            if name in self.src:
                raise SystemExit(f"{name}: in the source and built from --readout; pick one")

    def _src_experts(self, prefix):
        """Fused [E, 2I, H] gate_up and [E, H, I] down from a fused or per-expert source."""
        if f"{prefix}.gate_up_proj" in self.src:
            return self.src.get(f"{prefix}.gate_up_proj"), self.src.get(f"{prefix}.down_proj")
        n = 0
        while f"{prefix}.{n}.gate_proj.weight" in self.src:
            n += 1
        if not n:
            raise KeyError(f"no experts under {prefix} in source")
        gu = torch.stack([torch.cat([self.src.get(f"{prefix}.{e}.gate_proj.weight"),
                                     self.src.get(f"{prefix}.{e}.up_proj.weight")]) for e in range(n)])
        dn = torch.stack([self.src.get(f"{prefix}.{e}.down_proj.weight") for e in range(n)])
        return gu, dn

    def _experts(self, prefix):
        gu, dn = self._src_experts(prefix)
        out = {}
        inter = gu.shape[1] // 2
        for proj, w in (("gate_up", gu), ("down", dn)):
            w = w.to(self.dev)
            wsf2 = w.abs().amax(dim=(-2, -1)).float() / (FP4_MAX * FP8_MAX)  # one quantizer per expert
            packed, scale = nvfp4_quantize(w, wsf2[:, None, None])
            del w
            parts = ({"gate_proj": slice(0, inter), "up_proj": slice(inter, None)}
                     if proj == "gate_up" else {"down_proj": slice(None)})
            for e in range(packed.shape[0]):
                for name, rows in parts.items():
                    m = f"{prefix}.{e}.{name}"
                    out[f"{m}.weight"] = packed[e, rows].cpu()
                    out[f"{m}.weight_scale"] = scale[e, rows].cpu()
                    out[f"{m}.weight_scale_2"] = wsf2[e].cpu()
        return out

    def _module(self, module, algo):
        if algo == "FP8":
            q, s = fp8_quantize(self.src.get(f"{module}.weight").to(self.dev))
            return {f"{module}.weight": q.cpu(), f"{module}.weight_scale": s.cpu()}
        # ModelOpt fuses an MLP's gate_proj and up_proj for export: one weight_scale_2
        # from the larger amax of the two.
        head, _, leaf = module.rpartition(".")
        group = [module]
        if leaf in FUSED_GATE_UP:
            group = [f"{head}.{x}" for x in FUSED_GATE_UP if f"{head}.{x}" in self.layers]
        ws = {m: self.src.get(f"{m}.weight").to(self.dev) for m in group}
        wsf2 = torch.stack([amax(w) for w in ws.values()]).amax() / (FP4_MAX * FP8_MAX)
        out = {}
        for m, w in ws.items():
            packed, scale = nvfp4_quantize(w, wsf2)
            out.update({f"{m}.weight": packed.cpu(), f"{m}.weight_scale": scale.cpu(),
                        f"{m}.weight_scale_2": wsf2.cpu()})
        return out

    def _unit(self, name):
        """The tensors computed together with `name`."""
        if name.endswith(".input_scale"):
            return {name: self.ref.get(name).clone()}
        if name in self.synth:
            return {name: self.synth.pop(name)}
        m = EXPERT_RE.match(name)
        if m and m.group(1) in self.layers:
            return self._experts(m.group(1))
        module, _, suffix = name.rpartition(".")
        if module in self.layers and suffix in ("weight", "weight_scale", "weight_scale_2"):
            return self._module(module, self.layers[module]["quant_algo"])
        if name in self.src:
            return {name: self.src.get(name)}
        m = re.match(r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$", name)
        if m:  # unquantized fused experts (MTP) from a per-expert source
            gu, dn = self._src_experts(m.group(1))
            return {f"{m.group(1)}.gate_up_proj": gu, f"{m.group(1)}.down_proj": dn}
        raise KeyError(f"{name}: not in source and not derivable")

    def get(self, name):
        if name not in self.cache:
            self.cache.update(self._unit(name))
        return self.cache.pop(name)


def counts_as_parameter(name):
    """ModelOpt's index `total_parameters` is the in-memory HF model's num_parameters():
    scale siblings are buffers, and the MTP head is not part of the HF model."""
    return not name.startswith("mtp.") and not name.endswith(("weight_scale", "weight_scale_2", "input_scale"))


def unquantize(weight_map, hf_quant, quant_config, modules):
    """Drop `modules` from the quantized set: their weight stays where the reference puts
    it (copied from the source), their scale siblings go, and they join the ignore lists."""
    for m in modules:
        if m not in hf_quant["quantization"]["quantized_layers"]:
            raise SystemExit(f"--keep-bf16 {m}: not a quantized module of the reference")
        for suffix in ("weight_scale", "weight_scale_2", "input_scale"):
            weight_map.pop(f"{m}.{suffix}", None)
        hf_quant["quantization"]["quantized_layers"].pop(m)
        hf_quant["quantization"]["exclude_modules"].append(m)
        quant_config["quantized_layers"].pop(m)
        quant_config["ignore"].append(m)
        for group in quant_config["config_groups"].values():
            if m in group["targets"]:
                group["targets"].remove(m)


def convert(args):
    ref_index = load_json(os.path.join(args.ref, INDEX))
    weight_map = ref_index["weight_map"]
    hf_quant = load_json(os.path.join(args.ref, "hf_quant_config.json"))
    ref_config = load_json(os.path.join(args.ref, "config.json"))
    quant_config = ref_config["quantization_config"]
    config = load_json(os.path.join(args.src, "config.json"))
    text = config.get("text_config", config)
    synth = {}
    if args.readout:
        if "lm_head" not in args.keep_bf16:
            args.keep_bf16.append("lm_head")
        synth["lm_head.weight"], abs_err, rel_err = readout_lm_head(
            args.readout, text["vocab_size"], getattr(torch, text.get("dtype", "bfloat16")))
        print(f"lm_head from readout: max |cast error| {abs_err:.3e}, max relative {rel_err:.3e}", flush=True)
    unquantize(weight_map, hf_quant, quant_config, args.keep_bf16)
    if args.drop_mtp:
        mtp = [n for n in weight_map if n.startswith("mtp.")]
        if not mtp:
            raise SystemExit("--drop-mtp: the reference has no MTP head")
        for n in mtp:
            weight_map.pop(n)
        text["mtp_num_hidden_layers"] = 0
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free
    need = sum(os.path.getsize(os.path.join(args.ref, f)) for f in set(weight_map.values()))
    if free < need * 1.05:
        raise SystemExit(f"not enough disk: need ~{need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    os.makedirs(args.out, exist_ok=True)
    conv = Converter(args.src, args.ref, args.device, args.keep_bf16, synth)
    shards = {}
    for name, f in weight_map.items():
        shards.setdefault(f, []).append(name)
    total_size = total_params = 0
    t0 = time.time()
    for f in sorted(shards):
        tensors = {}
        for name in sorted(shards[f]):
            tensors[name] = conv.get(name).contiguous()
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        total_params += sum(t.numel() for n, t in tensors.items() if counts_as_parameter(n))
        save_file(tensors, os.path.join(args.out, f), metadata={"format": "pt"})
        print(f"{f}: {len(tensors)} tensors, {time.time() - t0:.0f}s", flush=True)
        del tensors
    if conv.cache:
        raise SystemExit(f"{len(conv.cache)} computed tensors not in the reference, e.g. {next(iter(conv.cache))}")
    unused = sorted(set(conv.src.where) - conv.src.used)
    if unused:
        raise SystemExit(f"{len(unused)} source tensors unused, e.g. {unused[:5]}")
    meta = {k: v for k, v in ref_index["metadata"].items()}
    meta.update({"total_parameters": total_params, "total_size": total_size} if "total_parameters" in meta
                else {"total_size": total_size})
    write_json(os.path.join(args.out, INDEX), {"metadata": meta, "weight_map": dict(sorted(weight_map.items()))})
    if args.keep_bf16:
        write_json(os.path.join(args.out, "hf_quant_config.json"), hf_quant)
    else:  # verbatim
        shutil.copyfile(os.path.join(args.ref, "hf_quant_config.json"), os.path.join(args.out, "hf_quant_config.json"))
    config["architectures"] = ref_config["architectures"]  # the tensor names are the reference's
    for k in REMOTE_CODE_KEYS:
        config.pop(k, None)
    config.pop("quantization_config", None)
    config["quantization_config"] = quant_config
    write_json(os.path.join(args.out, "config.json"), config)
    for f in AUX_FILES:
        if os.path.exists(os.path.join(args.src, f)):
            shutil.copyfile(os.path.join(args.src, f), os.path.join(args.out, f))
    print(f"done: {total_size / 1e9:.2f} GB, {total_params} params, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- comparison
def compare(args):
    a, b = Checkpoint(args.a, args.device), Checkpoint(args.b, args.device)
    names = sorted(set(a.where) | set(b.where))
    only = [n for n in names if (n in a) != (n in b)]
    same = diff = 0
    worst = []
    for n in names:
        if n in only:
            continue
        x, y = a.get(n), b.get(n)
        if x.dtype != y.dtype or x.shape != y.shape:
            print(f"MISMATCH-META {n}: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}")
            diff += 1
            continue
        xb, yb = x.contiguous().view(-1).view(torch.uint8), y.contiguous().view(-1).view(torch.uint8)
        if torch.equal(xb, yb):
            same += 1
            continue
        diff += 1
        frac = (xb != yb).float().mean().item()
        err = dequant_err(a, b, n, x, y)
        worst.append((err, n))
        print(f"DIFF {n}: {frac:.3e} of bytes differ, max |dequant diff| {err:.3e}", flush=True)
    for n in only:
        print(f"ONLY-IN-{'A' if n in a else 'B'} {n}")
    print(f"identical {same}/{same + diff} tensors ({same / max(1, same + diff):.6f}); "
          f"{len(only)} present on one side only")
    return 0 if not diff and not only else 1


def dequant_err(a, b, n, x, y):
    module, _, suffix = n.rpartition(".")
    try:
        if x.dtype == torch.uint8 and suffix == "weight":
            dx = nvfp4_dequantize(x, a.get(f"{module}.weight_scale"), a.get(f"{module}.weight_scale_2").float())
            dy = nvfp4_dequantize(y, b.get(f"{module}.weight_scale"), b.get(f"{module}.weight_scale_2").float())
        elif x.dtype == torch.float8_e4m3fn and suffix == "weight":
            dx = x.float() * a.get(f"{module}.weight_scale").float()
            dy = y.float() * b.get(f"{module}.weight_scale").float()
        else:
            dx, dy = x.float(), y.float()
        return (dx - dy).abs().max().item()
    except KeyError:
        return float("nan")


# ---------------------------------------------------------------- validation
def quantizer_signature(ck, module, algo):
    """What ModelOpt max calibration implies for one module: the largest per-block scale
    (NVFP4) or the largest code (FP8) is 448, i.e. the block holding amax(W) maps it to
    the top of the FP4/FP8 range."""
    w = ck.get(f"{module}.weight")
    if algo == "FP8":
        return w.float().abs().max().item()
    return ck.get(f"{module}.weight_scale").float().max().item()


def dequantize(ck, module, algo):
    w = ck.get(f"{module}.weight")
    if algo == "FP8":
        return w.float() * ck.get(f"{module}.weight_scale").float()
    return nvfp4_dequantize(w, ck.get(f"{module}.weight_scale"), ck.get(f"{module}.weight_scale_2").float())


def validate(args):
    """Exit 0 when OUT has the reference's structure (lm_head and MTP aside, as converted),
    matches the source within quantization error and passes the lm_head / input_scale checks."""
    ref, out, src = Checkpoint(args.ref), Checkpoint(args.out), Checkpoint(args.src, args.device)
    layers = load_json(os.path.join(args.out, "hf_quant_config.json"))["quantization"]["quantized_layers"]
    ref_layers = load_json(os.path.join(args.ref, "hf_quant_config.json"))["quantization"]["quantized_layers"]
    problems = []
    meta = {}
    for name in sorted(set(ref.where) | set(out.where)):
        if name not in ref.where or name not in out.where:
            meta.setdefault("only in " + ("ref" if name in ref.where else "out"), []).append(name)
            continue
        with safe_open(os.path.join(args.ref, ref.where[name]), "pt") as h:
            rs = h.get_slice(name)
            rmeta = (rs.get_dtype(), tuple(rs.get_shape()))
        with safe_open(os.path.join(args.out, out.where[name]), "pt") as h:
            os_ = h.get_slice(name)
            ometa = (os_.get_dtype(), tuple(os_.get_shape()))
        if rmeta != ometa:
            meta.setdefault("dtype/shape differs", []).append(f"{name}: ref {rmeta} out {ometa}")
    for k, v in meta.items():
        print(f"{k}: {len(v)}  e.g. {v[:4]}")
    allowed_only_ref = all(n.startswith("mtp.") or n.startswith("lm_head.weight_scale") for n in meta.get("only in ref", []))
    allowed_diff = all(n.startswith("lm_head.weight:") for n in meta.get("dtype/shape differs", []))
    if meta.get("only in out") or not allowed_only_ref or not allowed_diff:
        problems.append("structure")
    same_algo = {m: q for m, q in layers.items() if ref_layers.get(m, {}).get("quant_algo") == q["quant_algo"]}
    if len(same_algo) != len(layers):
        problems.append("quantized_layers differ from the reference")
    print(f"quantized modules: {len(layers)} (reference {len(ref_layers)}), same algorithm {len(same_algo)}")
    # input_scale: copied from the reference
    scales = [n for n in out.where if n.endswith(".input_scale")]
    bad = [n for n in scales if not torch.equal(out.get(n), ref.get(n))]
    print(f"input_scale tensors identical to the reference: {len(scales) - len(bad)}/{len(scales)}")
    problems += ["input_scale"] if bad else []
    # quantizer signature: 448 at the top of every module, for both checkpoints
    for label, ck in (("out", out), ("ref", ref)):
        sig = {}
        for m, q in layers.items():
            v = quantizer_signature(ck, m, q["quant_algo"])
            sig.setdefault(q["quant_algo"], []).append(v)
        for algo, vals in sig.items():
            top = sum(v == FP8_MAX for v in vals)
            print(f"{label} {algo}: max scale/code == 448 in {top}/{len(vals)} modules "
                  f"(min {min(vals):.4g}, max {max(vals):.4g})")
    # NVFP4 gate/up share weight_scale_2 in both
    for label, ck in (("out", out), ("ref", ref)):
        pairs = [m[: -len("gate_proj")] for m in layers if m.endswith("mlp.gate_proj")]
        shared = sum(torch.equal(ck.get(p + "gate_proj.weight_scale_2"), ck.get(p + "up_proj.weight_scale_2")) for p in pairs)
        print(f"{label}: gate/up share weight_scale_2 in {shared}/{len(pairs)} MLPs")
    # dequantization error against the source, on a sample
    sample = sorted(layers)[:: max(1, len(layers) // args.samples)]
    worst = {}
    for m in sample:
        algo = layers[m]["quant_algo"]
        w = src.get(f"{m}.weight").float()
        d = dequantize(out, m, algo).to(w.device)
        rel = ((d - w).norm() / w.norm()).item()
        worst.setdefault(algo, []).append(rel)
        dr = dequantize(ref, m, algo).to(w.device)  # the reference's own model, not this source
        worst.setdefault(algo + ", reference vs source", []).append(((dr - w).norm() / w.norm()).item())
        worst.setdefault(algo + ", out vs reference", []).append(((d - dr).norm() / dr.norm()).item())
    for k, v in worst.items():
        print(f"relative Frobenius error, {k}: mean {sum(v) / len(v):.4e}, max {max(v):.4e} over {len(v)} modules")
    if max(worst.get("W4A16_NVFP4", [0]) + worst.get("FP8", [0])) > 0.2:
        problems.append("dequantization error")
    # unquantized tensors are the source's
    plain = [n for n in out.where if n in src and n != "lm_head.weight"
             and not n.endswith(("weight_scale", "weight_scale_2", "input_scale"))
             and n.rpartition(".")[0] not in layers]
    differ = [n for n in plain if not torch.equal(out.get(n), src.get(n).to("cpu"))]
    print(f"unquantized tensors identical to the source: {len(plain) - len(differ)}/{len(plain)}")
    problems += ["unquantized tensors"] if differ else []
    if args.readout:
        cfg = load_json(os.path.join(args.out, "config.json"))
        text = cfg.get("text_config", cfg)
        want, abs_err, rel_err = readout_lm_head(args.readout, text["vocab_size"], getattr(torch, text.get("dtype", "bfloat16")))
        ok = torch.equal(out.get("lm_head.weight"), want)
        print(f"lm_head: equals the readout cast (rows at token_ids, zeros elsewhere): {ok}; "
              f"cast max abs {abs_err:.3e}, max rel {rel_err:.3e}")
        problems += [] if ok else ["lm_head"]
    print("VALID" if not problems else f"INVALID: {problems}")
    return 1 if problems else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--src", required=True, help="bf16 checkpoint directory")
    c.add_argument("--ref", required=True, help="ModelOpt reference checkpoint directory (format template)")
    c.add_argument("--out", required=True)
    c.add_argument("--keep-bf16", action="append", default=[], metavar="MODULE",
                   help="leave this reference-quantized module in the source dtype and list it as "
                        "excluded (repeatable), e.g. lm_head")
    c.add_argument("--readout", metavar="DIR",
                   help="build lm_head from DIR/readout.safetensors at DIR/decision_config.json token_ids "
                        "(a code-readout decision model); implies --keep-bf16 lm_head")
    c.add_argument("--drop-mtp", action="store_true",
                   help="leave out the reference's MTP head (a source without one)")
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    v = sub.add_parser("validate")
    v.add_argument("--src", required=True)
    v.add_argument("--ref", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--readout", metavar="DIR")
    v.add_argument("--samples", type=int, default=40, help="quantized modules to dequantize")
    v.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    k = sub.add_parser("compare")
    k.add_argument("a")
    k.add_argument("b")
    k.add_argument("--device", default="cpu")
    args = p.parse_args()
    sys.exit({"convert": convert, "compare": compare, "validate": validate}[args.cmd](args))


if __name__ == "__main__":
    main()
