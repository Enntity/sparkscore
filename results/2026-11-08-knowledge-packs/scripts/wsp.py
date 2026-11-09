"""Helpers for weight-space edits on d3-ultra BF16 (d3's own runtime)."""
import contextlib, json, math, os, sys

MODEL = "/w/models/d3-ultra"
S = {}  # m, lm, torch, originals


def boot():
    sys.path.insert(0, MODEL)
    import torch
    from d3_runtime import D3

    m = D3.from_pretrained(MODEL, device="cuda:0", verify="none")
    lm = m.backbone.language_model
    S.update(m=m, lm=lm, torch=torch, orig={})
    return dict(m=m, lm=lm, torch=torch, S=S)


# ---------------------------------------------------------------- scoring

def ask(state, questions):
    """d3's own answers (system_one): {name: probabilities list in option order}."""
    out = S["m"].system_one(state=state, questions=questions)["answers"]
    res = {}
    for k, a in out.items():
        q = questions[k]
        if q["type"] == "noul":
            res[k] = [1 - a["noul"], a["noul"]]
        elif q["type"] == "choice":
            res[k] = [a["probabilities"][c] for c in q["criteria"]]
        else:
            res[k] = [a["probabilities"][str(i)] for i in range(len(q["criteria"]))]
    return res


def rows(state, questions):
    """(name, token ids, state block tokens, option count) per question: the reference (one row per question) path."""
    p = S["m"].prepare(state, questions)
    return [(k, p.sequences[k], p.states.get(k), len(p.questions[k].keys)) for k in p.runnable]


# ---------------------------------------------------------------- capture

@contextlib.contextmanager
def capture(layers, keys=True, resid=True):
    """Record, for batch row 0, the down_proj input (MLP key, [T, 17408]) and the layer output ([T, 5120])."""
    lm, torch = S["lm"], S["torch"]
    rec = {"k": {}, "r": {}}
    hs = []
    for l in layers:
        layer = lm.layers[l]
        if keys:
            hs.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda mod, inp, l=l: rec["k"].__setitem__(l, inp[0][0].float())))
        if resid:
            hs.append(layer.register_forward_hook(
                lambda mod, inp, out, l=l: rec["r"].__setitem__(l, (out[0] if isinstance(out, tuple) else out)[0].float())))
    try:
        yield rec
    finally:
        for h in hs:
            h.remove()


def run_row(seq, nstate, count):
    """Code probabilities of one row at temperature 1 (logits through PromptEncoder with prefix_bidir blocks)."""
    torch = S["torch"]
    z = S["m"].logits([seq], [count], [nstate] if nstate is not None else None)[0, :count]
    return torch.softmax(z / S["m"].temperature, -1).tolist()


# ---------------------------------------------------------------- edits

def W(l):
    return S["lm"].layers[l].mlp.down_proj.weight


def save(l):
    if l not in S["orig"]:
        S["orig"][l] = W(l).detach().clone()


def add_delta(l, delta):
    save(l)
    with S["torch"].no_grad():
        W(l).add_(delta.to(W(l).dtype))


def restore():
    with S["torch"].no_grad():
        for l, w in S["orig"].items():
            W(l).copy_(w)
    S["orig"].clear()


def kl(p, q):
    return sum(a * math.log(max(a, 1e-12) / max(b, 1e-12)) for a, b in zip(p, q))


def nvfp4_fq(w):
    """Fake NVFP4 (as tools/nvfp4_convert.py): per-tensor wsf2 = amax / (6 * 448), FP8 E4M3 scale per 16, E2M1 values."""
    torch = S["torch"]
    w = w.float()
    wsf2 = w.abs().amax() / (6.0 * 448.0)
    b = w.view(w.shape[0], -1, 16)
    s = b.abs().amax(-1, keepdim=True) / (6.0 * wsf2)
    s[s == 0] = 1.0
    s = s.to(torch.float8_e4m3fn).float()
    x = b / (s * wsf2)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=w.device)
    idx = torch.bucketize(x.abs(), (grid[1:] + grid[:-1]) / 2)
    return (x.sign() * grid[idx] * s * wsf2).view_as(w)


def set_weight(l, value):
    save(l)
    with S["torch"].no_grad():
        W(l).copy_(value.to(W(l).dtype))


@contextlib.contextmanager
def patch(l, positions, values):
    """Replace layer l's output (batch row 0) at `positions` with `values` [len(positions), 5120]."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone(); h[0, positions] = values.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hd = S["lm"].layers[l].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


@contextlib.contextmanager
def steer(l, start, vec):
    """Add `vec` [5120] to layer l's output at positions >= start (batch row 0)."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone(); h[0, start:] += vec.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hd = S["lm"].layers[l].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()
