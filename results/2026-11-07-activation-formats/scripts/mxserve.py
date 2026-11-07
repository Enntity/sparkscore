# SPDX-License-Identifier: AGPL-3.0-only
"""d3's own server with fake-quantized weights: python mxserve.py MODEL_DIR CONVERTER_DIR WEIGHTS ACTS -- <d3_server args>"""
import os, sys
model_dir, conv_dir, variant, act = sys.argv[1:5]
argv = sys.argv[sys.argv.index("--") + 1:]
sys.path[:0] = [model_dir, conv_dir, os.path.dirname(os.path.abspath(__file__))]
import d3_runtime, mxquant
orig = d3_runtime.D3.from_pretrained.__func__
def patched(cls, *a, **kw):
    m = orig(cls, *a, **kw)
    v = variant
    print("MXQ", v, "relative weight RMS error:", {k: round(x, 4) for k, x in mxquant.apply(m, v).items()}, flush=True)
    mxquant.install_act_hooks(m); mxquant.ACT["mode"] = act
    print("MXQ acts", act, flush=True)
    return m
d3_runtime.D3.from_pretrained = classmethod(patched)
import d3_server
sys.exit(d3_server.main(argv))
