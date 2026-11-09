# G1: d3 BF16 reference for knowledge packs. Per item: probabilities with the knowledge as an isolated system prefix
# (what a pack computes), in the state, and absent. Written to /w/d3work/wsp/pack-reference.jsonl.
import importlib, sys, json, torch, rules4 as R, handbook as H, compile as cp, wsp, d3_runtime
S0 = dict(S)
for mod in (wsp, cp, R, H): importlib.reload(mod)
wsp.S.update(S0); S = wsp.S
from transformers.modeling_utils import AttentionInterface
fmt = sys.modules["d3_format"]; SYS0 = fmt.SYSTEM_PROMPT; orig = d3_runtime.block_attention
END = [None]
def isolated(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    out, w = orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    for row, (first, _) in enumerate(attention_mask.bounds):
        p1 = first + END[0]; g = query.shape[1] // key.shape[1]
        k = key[row:row+1, :, first:p1].repeat_interleave(g, 1); v = value[row:row+1, :, first:p1].repeat_interleave(g, 1)
        out[row:row+1, first:p1] = torch.nn.functional.scaled_dot_product_attention(query[row:row+1, :, first:p1], k, v, scale=scaling).transpose(1, 2)
    return out, w
temp = S["m"].temperature_for(3)
def prefix_probs(state, knowledge, qs):
    fmt.SYSTEM_PROMPT = SYS0
    _, s0, _, _ = wsp.rows(state, qs)[0]
    fmt.SYSTEM_PROMPT = SYS0 + "\n\n" + knowledge
    try:
        rs = wsp.rows(state, qs)
        m, a = cp.align(s0, rs[0][1]); END[0] = a + len(rs[0][1]) - len(s0)
        AttentionInterface.register(d3_runtime.BLOCK_ATTENTION, isolated)
        z = S["m"].logits([r[1] for r in rs], [r[3] for r in rs], [r[2] for r in rs]) / temp
        return {r[0]: torch.softmax(z[i, :r[3]], -1).tolist() for i, r in enumerate(rs)}, END[0]
    finally:
        fmt.SYSTEM_PROMPT = SYS0
        AttentionInterface.register(d3_runtime.BLOCK_ATTENTION, orig)
sets = [("handbook", H.POLICY, H.QS, H.build_test(0, 5)), ("rules4", R.POLICY, R.QS, R.build(0, "test", combos=True))]
with open("/w/d3work/wsp/pack-reference.jsonl", "w") as f:
    for name, know, qs, items in sets:
        for j, it in enumerate(items):
            pp, end = prefix_probs(it["state"], know, qs)
            row = dict(id=f"{name}-{j}", set=name, knowledge=know, state=it["state"], questions=qs, triggers=list(it["triggers"]),
                       gold=it["gold"], prefix_end_tokens=end, probs_prefix=pp,
                       probs_instate=wsp.ask(know + "\n\n" + it["state"], qs), probs_none=wsp.ask(it["state"], qs))
            f.write(json.dumps(row) + "\n")
        print(name, len(items), "items; prefix tokens", end, flush=True)
