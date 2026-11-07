# SPDX-License-Identifier: AGPL-3.0-only
"""Option probabilities -> Jev answer fields.

The fields and formulas follow TypeSafe's system-one-adapter-python 0.2.1
(github.com/typesafe-ai/system-one-adapter-python, tag v0.2.1, MIT):
_utils/confidence_metrics.py for the two confidence formulas and
_client.py::_convert_llm_value_to_typesafe_answer for the answer shapes.
They are reimplemented here, not copied. The same formulas are published at
docs.typesafe.ai/confidence and used by SGLang's /v1/systemone.

- noul:   {"noul": P(true)}
- choice: choice = the first key with the highest probability;
          confidence = (p_max - 1/K) / (1 - 1/K), 1 when K = 1
- score:  score = sum_i i * p_i (on probabilities rescaled to sum to 1);
          confidence = max(0, 1 - sum_i p_i |i - m| / MAD_unif), m the first
          most likely level, MAD_unif = mean_i |i - (K - 1)/2|, 1 when K = 1;
          legend = {"0": criteria[0], ...}; probabilities keyed "0".."K-1"

Both confidences normalise their input first, falling back to uniform when
the probabilities sum to 0, as the adapter does.
"""

import math


def _normalize(probs):
    total = math.fsum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def _first_argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def choice_confidence(probs):
    if len(probs) == 1:
        return 1.0
    q = _normalize(probs)
    uniform = 1.0 / len(q)
    return (max(q) - uniform) / (1.0 - uniform)


def score_confidence(probs):
    if len(probs) == 1:
        return 1.0
    q = _normalize(probs)
    n = len(q)
    mode = _first_argmax(q)
    spread = math.fsum(p * abs(i - mode) for i, p in enumerate(q))
    center = (n - 1) / 2
    uniform_spread = math.fsum(abs(i - center) for i in range(n)) / n
    return max(0.0, 1.0 - spread / uniform_spread)


def temper(probs, temperature):
    """Divide the option logits by `temperature`: p_i^(1/T), renormalised."""
    if temperature == 1.0:
        return _normalize(probs)
    logs = [math.log(p) / temperature if p > 0 else -math.inf for p in probs]
    peak = max(logs)
    if peak == -math.inf:
        return [1.0 / len(probs)] * len(probs)
    return _normalize([math.exp(x - peak) for x in logs])


def answer(question, keys, probs):
    """One answer object. `keys` and `probs` are in option order, as
    render.Branch.keys gives them; `probs` sums to 1."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probs[keys.index("true")]}
    by_key = dict(zip(keys, probs))
    if kind == "choice":
        return {
            "type": "choice",
            "choice": keys[_first_argmax(probs)],
            "confidence": choice_confidence(probs),
            "probabilities": by_key,
        }
    q = _normalize(probs)
    return {
        "type": "score",
        "score": math.fsum(i * p for i, p in enumerate(q)),
        "confidence": score_confidence(probs),
        "legend": {str(i): level for i, level in enumerate(question["criteria"])},
        "probabilities": by_key,
    }
