"""Context compilation into one MLP down_proj: closed-form least squares over (key -> residual delta) pairs.

For each training item, the residual at layer L with the policy in the state minus the residual without it is the
target at every aligned position; the MLP key at layer L without the policy is the input. Generic keys from d3's
reference requests anchor the solve (they map to no change). No optimizer: forward passes and one linear solve.
"""
import tickets
from wsp import S, rows, capture, add_delta

def align(s0, s1):
    """Positions of s0 in s1 when s1 = s0 with tokens inserted at one place; None if not that shape."""
    a = 0
    while a < len(s0) and s0[a] == s1[a]: a += 1
    d = len(s1) - len(s0)
    if s1[a + d:] != s0[a:]: return None
    return list(range(a)) + [j + d for j in range(a, len(s0))], a

def forward(seq, ns, count, layers):
    with capture(layers) as rec:
        S["m"].logits([seq], [count], [ns])
    return rec

def pairs(items, L, positions="all", q=None):
    """Stacked keys K [N, 17408] and targets R [N, 5120] at layer L."""
    torch = S["torch"]; q = q or tickets.TEAM_Q
    Ks, Rs = [], []
    for it in items:
        (_, s0, n0, c), = rows(it["state"], q)
        (_, s1, n1, _), = rows(tickets.with_policy(it["state"]), q)
        m, a = align(s0, s1)
        r0 = forward(s0, n0, c, [L]); r1 = forward(s1, n1, c, [L])
        k, d = r0["k"][L], r1["r"][L][m] - r0["r"][L]
        sel = {"all": slice(None), "last": slice(-1, None), "ticket": slice(a, None)}[positions]
        Ks.append(k[sel]); Rs.append(d[sel])
    return torch.cat(Ks), torch.cat(Rs)

_generic = {}
def generic_moment(L, COLL):
    """Sum of k k^T over every position of d3's reference requests at layer L, and the count."""
    if L in _generic: return _generic[L]
    torch = S["torch"]; C, n = 0, 0
    for state, qs, _ in COLL:
        for _, s, ns, c in rows(state, qs)[:1]:
            k = forward(s, ns, c, [L])["k"][L]
            C = C + k.T @ k; n += k.shape[0]
    _generic[L] = (C, n)
    return C, n

def solve(K, R, C0, n0, lam, eps=1e-4):
    """Delta [5120, 17408] minimising |K D^T - R|^2 + lam * (N / n0) * sum_generic |D k|^2 + eps-ridge."""
    torch = S["torch"]
    N = K.shape[0]
    A = K.T @ K + (lam * N / n0) * C0
    A += eps * A.diagonal().mean() * torch.eye(A.shape[0], device=A.device)
    X = torch.linalg.solve(A, K.T @ R)   # [17408, 5120]
    return X.T.contiguous()


def pairs_q(items, L, q=None, zero_controls=True):
    """Keys and targets at the question-block positions only; control items (no Helios) target no change."""
    torch = S["torch"]; q = q or tickets.TEAM_Q
    Ks, Rs = [], []
    for it in items:
        (_, s0, n0, c), = rows(it["state"], q)
        r0 = forward(s0, n0, c, [L])
        k = r0["k"][L][n0:]
        if zero_controls and it["product"] != "Helios":
            d = torch.zeros(k.shape[0], r0["r"][L].shape[1], device=k.device)
        else:
            (_, s1, n1, _), = rows(tickets.with_policy(it["state"]), q)
            m, a = align(s0, s1)
            d = forward(s1, n1, c, [L])["r"][L][m][n0:] - r0["r"][L][n0:]
        Ks.append(k); Rs.append(d)
    return torch.cat(Ks), torch.cat(Rs)
