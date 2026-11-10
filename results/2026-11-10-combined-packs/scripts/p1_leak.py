# SPDX-License-Identifier: AGPL-3.0-only
"""Which answers moved, per condition, in raw/combo.jsonl: the question, and how often other tickets got "Can wait".

Usage: python scripts/p1_leak.py raw/combo.jsonl
"""
import json, sys

rows = [json.loads(line) for line in open(sys.argv[1])]
arg = lambda p: max(range(len(p)), key=p.__getitem__)
other = [r for r in rows if not {"osprey", "kestrel"} & set(r["triggers"])]
print(f"{'condition':<17} {'changed vs reference':<26} {'other tickets: Can wait':>23} {'mean p(Can wait)':>17}")
for cond in rows[0]["probs"]:
    ref = "in-state-rev" if cond.endswith("-rev") else "in-state"
    moved = {}
    for r in rows:
        for q, p in r["probs"][cond].items():
            if arg(p) != arg(r["probs"][ref][q]):
                moved[q] = moved.get(q, 0) + 1
    wait = sum(arg(r["probs"][cond]["urgency"]) == 0 for r in other)
    mean = sum(r["probs"][cond]["urgency"][0] for r in other) / len(other)
    print(f"{cond:<17} {json.dumps(moved):<26} {wait:>19}/{len(other)} {mean:>17.2f}")
