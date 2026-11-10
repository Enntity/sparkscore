# P1: can knowledge packs built apart be read together? d3-ultra BF16, d3's own layers (StateOnce's code path).
#
# The 7-rule handbook is split into two documents: A (sections 1-7: wisteria, helios, kestrel, lattice) and B
# (sections 8-13 under their own title: osprey, tamarind, orchard). Every condition reads the same tokens
# [sys | A | B | state | question]; only who reads whom, and the GDN state the request starts from, differ.
#
#   none           no knowledge (d3's own logits path)
#   in-state       A and B prepended to the state (d3's trained regime; the reference)
#   sys-bidir      A and B in the system prompt, d3's mask (sanity: our block runner must match d3's logits)
#   one-pack       sys+A+B one pack (today's engine with the combined text)
#   chain          sys+A, then B reading sys+A+B; exact GDN scan (pack tree)
#   chain-base     sys, then A reading sys, then B reading sys+A; exact GDN scan
#   iso-composed   sys, then A and B each reading sys only (B never sees A); the request's GDN entry state
#                  composes them: S = M_B S_A + (S_B - M_B S_sys), M_B from scanning B's features from I with v = 0
#   iso-naive      as iso-composed, but the request starts from B's own GDN state (A's dropped)
#   *-rev          B before A (order sensitivity)
import json, math, sys, time, torch
import handbook as H, d3_runtime, wsp
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

m, lm = S["m"], S["lm"]
fmt = sys.modules["d3_format"]
SYS0 = fmt.SYSTEM_PROMPT
dev = m.device
TEMP = m.temperature_for(3)
cfg = lm.config
STACK = [(layer, kind == "full_attention")
         for layer, kind in zip(lm.layers[:cfg.num_hidden_layers], cfg.layer_types[:cfg.num_hidden_layers])]
EMB = m.backbone.get_input_embeddings()
sdpa = torch.nn.functional.scaled_dot_product_attention

secs = H.POLICY.split("\n\n")
DOC_A = "\n\n".join(secs[:8])
DOC_B = "\n\n".join(["ACME CUSTOMER SUPPORT HANDBOOK, PART 2 (internal, version 14)"] + secs[8:])
WHERE = {"wisteria": "A", "helios": "A", "kestrel": "A", "lattice": "A", "osprey": "B", "tamarind": "B", "orchard": "B"}
for r, d in WHERE.items():
    assert (r.title() in DOC_A) == (d == "A") and (r.title() in DOC_B) == (d == "B"), r
ITEMS = H.build_test(0, 5)
QN = list(H.QS)


def prepared(sysprompt, state):
    fmt.SYSTEM_PROMPT = sysprompt
    try:
        return m.prepare(state, H.QS)
    finally:
        fmt.SYSTEM_PROMPT = SYS0


def prefix_len(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]: i += 1
    return i


def layout(first, second):
    """Token boundaries (k0, k1, k2) of [sys | first | second] in the system prompt, and that prompt."""
    full_sys = SYS0 + "\n\n" + first + "\n\n" + second
    full = prepared(full_sys, "x").sequences[QN[0]]
    one = prepared(SYS0 + "\n\n" + first, "x").sequences[QN[0]]
    base = prepared(SYS0, "x").sequences[QN[0]]
    k0, k1 = prefix_len(base, full), prefix_len(one, full)
    k2 = len(full) - prefix_len(base[::-1], full[::-1])
    assert 0 < k0 < k1 < k2, (k0, k1, k2)
    return full_sys, (k0, k1, k2)


# ------------------------------------------------------------------ block runner (StateOnce's layers, any reads)

def rope_at(emb, start):
    pos = torch.arange(start, start + emb.shape[1], device=dev).view(1, 1, -1).expand(3, 1, -1)
    return lm.rotary_emb(emb, pos)


def attn(a, h, rope, reads):
    shape = h.shape[:-1]
    heads = (*shape, -1, a.head_dim)
    q, gate = torch.chunk(a.q_proj(h).view(*shape, -1, a.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    q = a.q_norm(q.view(heads)).transpose(1, 2)
    k = a.k_norm(a.k_proj(h).view(heads)).transpose(1, 2)
    v = a.v_proj(h).view(heads).transpose(1, 2)
    cos, sin = rope
    q, k = qwen.apply_rotary_pos_emb(q, k, cos, sin)
    K = torch.cat([r[0] for r in reads] + [k], 2)
    V = torch.cat([r[1] for r in reads] + [v], 2)
    g = q.shape[1] // K.shape[1]
    if g > 1:
        K, V = K.repeat_interleave(g, 1), V.repeat_interleave(g, 1)
    o = sdpa(q, K, V, scale=a.scaling).transpose(1, 2).reshape(*shape, -1)
    return a.o_proj(o * torch.sigmoid(gate)), (k, v)


def delta(d, h, entry, want_m, alt):
    rows, L, _ = h.shape
    mixed = d.in_proj_qkv(h).transpose(1, 2)
    z = d.in_proj_z(h).reshape(rows, L, -1, d.head_v_dim)
    b, a = d.in_proj_b(h), d.in_proj_a(h)
    W = d.conv_kernel_size - 1
    tail0, initial = entry if entry is not None else (mixed.new_zeros(rows, mixed.shape[1], W), None)
    joined = torch.cat([tail0, mixed], -1)
    conv = qwen.causal_conv1d_fn(joined, d.conv1d.weight.squeeze(1), d.conv1d.bias, activation=d.activation)[:, :, W:]
    q, k, v = torch.split(conv.transpose(1, 2), [d.key_dim, d.key_dim, d.value_dim], dim=-1)
    q = q.reshape(rows, L, -1, d.head_k_dim)
    k = k.reshape(rows, L, -1, d.head_k_dim)
    v = v.reshape(rows, L, -1, d.head_v_dim)
    beta = b.sigmoid()
    g = -d.A_log.float().exp() * torch.nn.functional.softplus(a.float() + d.dt_bias)
    rep = d.num_v_heads // d.num_k_heads
    if rep > 1:
        q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
    scan = lambda vv, s0: qwen.torch_chunk_gated_delta_rule(q, k, vv, g=g, beta=beta, initial_state=s0,
                                                            output_final_state=True, use_qk_l2norm_in_kernel=True)
    o, final = scan(v, None if initial is None else initial.contiguous())
    e = {"tail": joined[:, :, -W:], "final": final}
    if want_m:  # transition: the scan of these features from I with v = 0, so final(S0) = M @ S0 + C
        K_ = d.head_k_dim
        eye = torch.eye(K_, device=dev, dtype=torch.float32).expand(rows, k.shape[2], K_, K_).contiguous()
        _, e["M"] = scan(torch.zeros_like(k), eye)
        e["from"] = initial if initial is not None else torch.zeros_like(final)
    if alt is not None:  # the same features scanned from another state: checks the composition
        _, e["alt"] = scan(v, alt.contiguous())
    o = d.norm(o.reshape(-1, d.head_v_dim), z.reshape(-1, d.head_v_dim))
    return d.out_proj(o.reshape(rows, L, -1)), e


def block(ids, start, reads, dfrom, want_m=False, alt=None):
    """Tokens `ids` at positions start.. . reads: earlier blocks whose keys the attention layers read (besides its own,
    both directions); dfrom: the per-layer GDN (tail, state) entry, or None. alt: per-layer states for the check."""
    hidden = EMB(torch.as_tensor(ids, device=dev)[None])
    rope = rope_at(hidden, start)
    out = []
    for i, (layer, softmax) in enumerate(STACK):
        res = hidden
        normed = layer.input_layernorm(hidden)
        if softmax:
            mixed, e = attn(layer.self_attn, normed, rope, [r[i] for r in reads])
        else:
            mixed, e = delta(layer.linear_attn, normed, None if dfrom is None else dfrom[i], want_m,
                             None if alt is None else alt[i])
        hidden = res + mixed
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        out.append(e)
    return out, hidden


def entry_of(blk):
    return [None if softmax else (blk[i]["tail"], blk[i]["final"]) for i, (_, softmax) in enumerate(STACK)]


def build(kind, toks, k0, k1, k2):
    """(reads, GDN entry) the request starts from, for a pack layout over toks[:k2]; and a composition check."""
    chk = None
    if kind == "one-pack":
        X, _ = block(toks[:k2], 0, [], None); return [X], entry_of(X), chk
    if kind == "chain":
        P1, _ = block(toks[:k1], 0, [], None)
        P2, _ = block(toks[k1:k2], k1, [P1], entry_of(P1)); return [P1, P2], entry_of(P2), chk
    B0, _ = block(toks[:k0], 0, [], None)
    A, _ = block(toks[k0:k1], k0, [B0], entry_of(B0))
    if kind == "chain-base":
        Bb, _ = block(toks[k1:k2], k1, [B0, A], entry_of(A)); return [B0, A, Bb], entry_of(Bb), chk
    alt = [None if softmax else A[i]["final"] for i, (_, softmax) in enumerate(STACK)]
    Bi, _ = block(toks[k1:k2], k1, [B0], entry_of(B0), want_m=True, alt=alt)
    ent, errs = [], []
    for i, (_, softmax) in enumerate(STACK):
        if softmax: ent.append(None); continue
        e = Bi[i]
        if kind == "iso-naive":
            ent.append((e["tail"], e["final"])); continue
        comp = e["M"] @ A[i]["final"] + (e["final"] - e["M"] @ e["from"])
        errs.append(((comp - e["alt"]).abs().max() / e["alt"].abs().max()).item())
        ent.append((e["tail"], comp))
    return [B0, A, Bi], ent, (max(errs) if errs else None)


def score_item(reads, ent, rows, k2):
    """Probabilities per question for one item's rows on a pack (reads, ent) covering [0, k2)."""
    (_, s0, n0, _) = rows[0]
    assert all(r[1][:n0] == s0[:n0] and r[2] == n0 for r in rows)
    st, _ = block(s0[k2:n0], k2, reads, ent)
    ent2 = entry_of(st)
    out = {}
    for name, seq, ns, count in rows:
        _, hid = block(seq[ns:], ns, reads + [st], ent2)
        z = (lm.norm(hid[:, -1]).float() @ m.readout.T)[0, :count] / TEMP
        out[name] = torch.softmax(z, -1).tolist()
    return out


def rows_of(p):
    return [(k, p.sequences[k], p.states.get(k), len(p.questions[k].keys)) for k in p.runnable]


def d3_probs(p):
    rs = rows_of(p)
    z = m.logits([r[1] for r in rs], [r[3] for r in rs], [r[2] for r in rs]) / TEMP
    return {r[0]: torch.softmax(z[i, :r[3]], -1).tolist() for i, r in enumerate(rs)}


# ------------------------------------------------------------------ run

t0 = time.time()
KEYS = {k: list(prepared(SYS0, "x").questions[k].keys) for k in QN}
print("question keys", KEYS, "temperature", TEMP, flush=True)
LAY = {"AB": layout(DOC_A, DOC_B), "BA": layout(DOC_B, DOC_A)}
print("layouts (k0, k1, k2):", {k: v[1] for k, v in LAY.items()}, flush=True)
RES = {}
with torch.inference_mode(), d3_runtime.sdpa_backends(dev):
    for tag, know in (("", DOC_A + "\n\n" + DOC_B), ("-rev", DOC_B + "\n\n" + DOC_A)):
        RES["in-state" + tag] = [d3_probs(prepared(SYS0, know + "\n\n" + it["state"])) for it in ITEMS]
    RES["none"] = [d3_probs(prepared(SYS0, it["state"])) for it in ITEMS]
    # Code order vs d3's own answers (criteria order; noul as [no, yes]) on the rule items: the gold indices assume equal.
    miss = [(j, q) for j, it in enumerate(ITEMS) if it["triggers"]
            for q, p in wsp.ask(DOC_A + "\n\n" + DOC_B + "\n\n" + it["state"], H.QS).items()
            if max(range(len(p)), key=p.__getitem__) != max(range(len(p)), key=RES["in-state"][j][q].__getitem__)]
    print("code order vs d3 answers, argmax mismatches:", miss, flush=True)
    full_sys, (k0, k1, k2) = LAY["AB"]
    RES["sys-bidir-d3"] = [d3_probs(prepared(full_sys, it["state"])) for it in ITEMS]
    RES["sys-bidir"] = []
    for it in ITEMS:  # the runner with d3's own mask: one state block [0, n), then the questions
        rs = rows_of(prepared(full_sys, it["state"]))
        (_, s0, n0, _) = rs[0]
        st, _ = block(s0[:n0], 0, [], None)
        o = {}
        for name, seq, ns, count in rs:
            _, hid = block(seq[ns:], ns, [st], entry_of(st))
            o[name] = torch.softmax((lm.norm(hid[:, -1]).float() @ m.readout.T)[0, :count] / TEMP, -1).tolist()
        RES["sys-bidir"].append(o)
    CHK = {}
    for lay, kinds in (("AB", ["one-pack", "chain", "chain-base", "iso-composed", "iso-naive"]),
                       ("BA", ["one-pack", "iso-composed"])):
        full_sys, (k0, k1, k2) = LAY[lay]
        PR = [rows_of(prepared(full_sys, it["state"])) for it in ITEMS]
        toks = PR[0][0][1][:k2]
        assert all(r[0][1][:k2] == toks for r in PR), "pack tokens differ across items"
        for kind in kinds:
            name = kind + ("" if lay == "AB" else "-rev")
            reads, ent, chk = build(kind, toks, k0, k1, k2)
            if chk is not None: CHK[name] = chk
            RES[name] = [score_item(reads, ent, rs, k2) for rs in PR]
            print(f"  {name} done ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ report

arg = lambda p: max(range(len(p)), key=p.__getitem__)
kl = lambda p, q: sum(a * math.log(max(a, 1e-12) / max(b, 1e-12)) for a, b in zip(p, q))
def gold_idx(q, it):
    g = it["gold"].get(q)
    if q == "team": return KEYS[q].index(g)
    return g
def rules(res):
    by = {"A": [0, 0], "B": [0, 0]}
    for it, r in zip(ITEMS, res):
        for t in it["triggers"]:
            q = H.RULE_Q[t]; c = by[WHERE[t]]
            c[0] += arg(r[q]) == gold_idx(q, it); c[1] += 1
    return by
def team_controls(res):
    return sum(arg(r["team"]) == gold_idx("team", it) for it, r in zip(ITEMS, res) if not it["triggers"]), \
        sum(1 for it in ITEMS if not it["triggers"])
def vs(res, ref):
    a = sum(arg(x[q]) == arg(y[q]) for x, y in zip(res, ref) for q in QN)
    kls = [kl(y[q], x[q]) for x, y in zip(res, ref) for q in QN]
    return a, len(kls), sum(kls) / len(kls), max(kls)

print(f"\n{'condition':<15} {'A rules':>8} {'B rules':>8} {'all':>6} {'no-rule team':>12} {'vs in-state':>12} {'mean KL':>8} {'max KL':>7}")
for name, res in RES.items():
    by = rules(res); tc = team_controls(res)
    ref = RES["in-state-rev"] if name.endswith("-rev") else RES["in-state"]
    a, n, mk, xk = vs(res, ref)
    print(f"{name:<15} {by['A'][0]:>4}/{by['A'][1]:<3} {by['B'][0]:>4}/{by['B'][1]:<3} {by['A'][0]+by['B'][0]:>3}/{by['A'][1]+by['B'][1]:<2} "
          f"{tc[0]:>8}/{tc[1]:<3} {a:>8}/{n:<3} {mk:>8.4f} {xk:>7.3f}", flush=True)
sb = vs(RES["sys-bidir"], RES["sys-bidir-d3"])
print(f"\nrunner sanity: sys-bidir (runner) vs sys-bidir (d3 logits): argmax {sb[0]}/{sb[1]}, mean KL {sb[2]:.2e}, max KL {sb[3]:.2e}")
print("composition check, max |M S_A + C - scan(S_A)| / max|scan(S_A)| over GDN layers:", {k: f"{v:.1e}" for k, v in CHK.items()})
for a_, b_ in (("iso-composed", "chain-base"), ("iso-composed", "one-pack"), ("chain", "one-pack"), ("iso-composed-rev", "iso-composed")):
    x = vs(RES[a_], RES[b_]); print(f"{a_} vs {b_}: argmax {x[0]}/{x[1]}, mean KL {x[2]:.4f}, max KL {x[3]:.3f}")
with open("/w/d3work/wsp/combo.jsonl", "w") as f:
    for j, it in enumerate(ITEMS):
        f.write(json.dumps(dict(item=j, state=it["state"], triggers=list(it["triggers"]), gold=it["gold"],
                                probs={k: v[j] for k, v in RES.items()})) + "\n")
print(f"[{time.time() - t0:.0f}s] wrote /w/d3work/wsp/combo.jsonl", flush=True)
