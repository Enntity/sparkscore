# SPDX-License-Identifier: AGPL-3.0-only
"""Send one typed request to a running SparkScore and check the answers.

    python3 install/smoke.py [http://127.0.0.1:8015]

Standard library only. Exits non-zero if the server is down, the answers have
the wrong shape, or an answer that any working decision model gets right is
wrong.
"""
import json
import os
import sys
import time
import urllib.request

REQUEST = {
    "model": "sparkscore-d3",
    "state": (
        "Refund policy: purchases can be refunded within 30 days. Annual plans are refunded pro rata.\n\n"
        "Support ticket from Mina C. (annual plan, bought 5 months ago):\n"
        '"Honestly we barely use it anymore. I\'d like a refund for the rest of the year please."'
    ),
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Billing", "infrastructure": "Infrastructure", "sales": "Sales"}},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["Can wait", "This week", "Today", "Right now"]},
        "refund": {"type": "noul", "instructions": "Is a refund allowed under the policy?"},
    },
}


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8015").rstrip("/")
    headers = {"content-type": "application/json"}
    if os.environ.get("SPARKSCORE_API_KEY"):
        headers["authorization"] = f"Bearer {os.environ['SPARKSCORE_API_KEY']}"
    req = urllib.request.Request(f"{base}/v1/systemone", json.dumps(REQUEST).encode(), headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.load(r)
        timing = r.headers.get("server-timing", "")
        passes = r.headers.get("x-sparkscore-passes")
    ms = (time.perf_counter() - t0) * 1000
    a = body["answers"]
    print(json.dumps(a, indent=2))
    print(f"round trip {ms:.0f} ms · Server-Timing: {timing} · passes: {passes}")
    assert a["team"]["choice"] == "billing", "team should be billing"
    assert a["refund"]["noul"] > 0.5, "an annual-plan refund is allowed pro rata"
    assert abs(sum(a["urgency"]["probabilities"].values()) - 1) < 1e-3, "urgency should be a distribution"
    print("smoke test passed")


if __name__ == "__main__":
    main()
