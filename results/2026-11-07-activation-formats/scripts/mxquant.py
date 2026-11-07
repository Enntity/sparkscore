# SPDX-License-Identifier: AGPL-3.0-only
"""Fake-quantize d3's 4-bit-path weights in place (see mxq.py for the variants)."""
import math, os, sys
import torch
import nvfp4_convert as nc

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1(x):
    """Round to the E2M1 grid (nearest, ties to even code), clamped to +-6."""
    codes = nc.e2m1_codes(x.clamp(-6.0, 6.0))
    lut = torch.cat([E2M1, -E2M1]).to(x.device)
    return lut[codes.long()]


def mx_blocks(b, mode):
    """b: [..., 32] fp32 -> dequantized MXFP4 values."""
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    lo = torch.floor(torch.log2(amax)) - 2          # OCP rule: may clip the block max
    if mode == "ocp":
        s = torch.exp2(lo); return e2m1(b / s) * s
    best, err = None, None
    for e in (lo, lo + 1):
        s = torch.exp2(e); q = e2m1(b / s) * s
        d = ((q - b) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, err = q, d
        else:
            take = d < err; best = torch.where(take, q, best); err = torch.where(take, d, err)
    return best


def mx(w, mode):
    n, k = w.shape
    x = w.float()
    if mode == "row":
        ramax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        best, err = None, None
        for j in range(8):
            r = ramax / 6.0 * 2.0 ** (j / 8.0)
            q = mx_blocks((x / r).view(n, k // 32, 32), "mse").view(n, k) * r
            d = ((q - x) ** 2).sum(-1, keepdim=True)
            if best is None:
                best, err = q, d
            else:
                take = d < err; best = torch.where(take, q, best); err = torch.where(take, d, err)
        return best
    return mx_blocks(x.view(n, k // 32, 32), mode).view(n, k)


def nvfp4(w, wsf2):
    packed, scale = nc.nvfp4_quantize(w, wsf2)
    return nc.nvfp4_dequantize(packed, scale, wsf2)


def fp8_then_nvfp4(w):
    q, s = nc.fp8_quantize(w)
    w8 = (q.float() * s).to(torch.bfloat16)
    return nvfp4(w8, nc.amax(w8) / (6.0 * 448.0))



def apply(m, variant):
    lm = m.backbone.language_model
    errs = {}
    with torch.no_grad():
        for i, layer in enumerate(lm.layers):
            mats = [("mlp.gate_proj", layer.mlp.gate_proj), ("mlp.up_proj", layer.mlp.up_proj), ("mlp.down_proj", layer.mlp.down_proj)]
            if hasattr(layer, "self_attn"):
                a = layer.self_attn
                mats += [(f"self_attn.{p}", getattr(a, p)) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
            if variant == "nvfp4":
                g, u = layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight
                shared = torch.maximum(nc.amax(g), nc.amax(u)) / (6.0 * 448.0)
            for name, mod in mats:
                w = mod.weight.data
                if variant == "bf16":
                    continue
                if variant == "nvfp4":
                    if name.startswith("mlp.down"):
                        q = nvfp4(w, nc.amax(w) / (6.0 * 448.0))
                    elif name.startswith("mlp."):
                        q = nvfp4(w, shared)
                    else:
                        q = fp8_then_nvfp4(w)
                else:
                    q = mx(w, {"mx_ocp": "ocp", "mx_mse": "mse", "mx_row": "row"}[variant])
                x = w.float()
                kind = name.split(".")[-1]
                e = errs.setdefault(kind, [0.0, 0.0])
                e[0] += ((q.float() - x) ** 2).sum().item(); e[1] += (x ** 2).sum().item()
                mod.weight.data.copy_(q.to(w.dtype))
                del q, x
            torch.cuda.empty_cache()
    return {k: math.sqrt(v[0] / v[1]) for k, v in errs.items()}


# ---------------------------------------------------------------- activations
E4M3 = torch.float8_e4m3fn
ACT = {"mode": "bf16"}


def _e4m3(x):
    return x.to(E4M3).float()


def _nvfp4_term(x, s2):
    """One NVFP4 term of x [..., K] with per-token scale s2 [..., 1]: block-16 E4M3 scales, E2M1 values."""
    b = x.view(*x.shape[:-1], x.shape[-1] // 16, 16)
    sb = _e4m3(b.abs().amax(-1, keepdim=True) / (6.0 * s2.unsqueeze(-1)))
    sb = torch.where(sb == 0, torch.ones_like(sb), sb)
    s = sb * s2.unsqueeze(-1)
    return (e2m1(b / s) * s).view(x.shape)


def act_quant(x, mode):
    if mode == "bf16":
        return x
    if mode == "e4m3tok":   # today's engine: E4M3 with one FP32 scale per token
        s = (x.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
        return _e4m3(x / s) * s
    if mode == "mx8":       # MXFP8: E4M3 with a UE8M0 scale per 32
        b = x.view(*x.shape[:-1], x.shape[-1] // 32, 32)
        s = torch.exp2(torch.ceil(torch.log2((b.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30))))
        return (_e4m3(b / s) * s).view(x.shape)
    s2 = (x.abs().amax(-1, keepdim=True) / (6.0 * 448.0)).clamp_min(1e-30)
    hi = _nvfp4_term(x, s2)
    if mode == "nvfp4":     # W4A4, one term
        return hi
    if mode == "nvfp4x2":   # two NVFP4 terms sharing the per-token scale: two FP4 MMAs into one accumulator
        return hi + _nvfp4_term(x - hi, s2)
    raise ValueError(mode)


def targets(m):
    for layer in m.backbone.language_model.layers:
        mods = [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]
        if hasattr(layer, "self_attn"):
            mods += [getattr(layer.self_attn, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        yield from mods


_FILE = {"path": None, "mtime": None, "mode": "bf16"}


def current_mode():
    """ACT["mode"], or with "file:PATH" the scheme named in PATH (re-read when it changes)."""
    mode = ACT["mode"]
    if not mode.startswith("file:"):
        return mode
    path = mode[5:]
    mt = os.stat(path).st_mtime_ns
    if (path, mt) != (_FILE["path"], _FILE["mtime"]):
        _FILE.update(path=path, mtime=mt, mode=open(path).read().strip())
        print("MXQ acts now", _FILE["mode"], flush=True)
    return _FILE["mode"]


def install_act_hooks(m):
    def pre(mod, args):
        x = args[0]
        return (act_quant(x.float(), current_mode()).to(x.dtype),) + tuple(args[1:])
    return [mod.register_forward_pre_hook(pre) for mod in targets(m)]
