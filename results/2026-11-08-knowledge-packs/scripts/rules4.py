"""Four private rules over three questions asked together (team choice, urgency score, legal noul).

  helios   : a ticket about Helios goes to Infrastructure, whatever it is about          (team)
  orchard  : a ticket about Orchard goes to Sales, whatever it is about                  (team)
  kestrel  : a Kestrel-tier customer's ticket is "Right now" urgent                       (urgency)
  tamarind : a ticket that mentions Project Tamarind needs legal review                  (legal)

Every ticket has a product, a customer tier and a project line. Single-trigger items carry one trigger; combo items
(test only) carry two. Templates are split into train and test halves as in tickets.py.
"""
import random
import tickets as T

POLICY = ("Acme routing policy. Helios is run by the Infrastructure team end to end: every ticket about Helios goes "
          "to Infrastructure, whatever it is about. Orchard is sold and supported by the Sales team: every ticket about "
          "Orchard goes to Sales, whatever it is about. Kestrel-tier customers have a contractual one-hour response: "
          "any ticket from a Kestrel-tier customer is Right now urgent. Project Tamarind is under a confidentiality "
          "order: any ticket that mentions Project Tamarind needs legal review.")

QS = {
    "team": T.TEAM_Q["team"],
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["Can wait", "This week", "Today", "Right now"]},
    "legal": {"type": "noul", "instructions": "Does this ticket need legal review?"},
}
RULE_Q = {"helios": "team", "orchard": "team", "kestrel": "urgency", "tamarind": "legal"}
RULES = list(RULE_Q)
TIERS = ["Sparrow", "Finch", "Wren", "Heron"]
PROJECTS = ["Juniper", "Bramble", "Saffron", "Marigold"]
PRODUCTS = [p for p in T.CONTROLS if p != "Orion"]


def _item(rng, kind, temp, default, triggers):
    product = "Helios" if "helios" in triggers else "Orchard" if "orchard" in triggers else rng.choice(PRODUCTS)
    tier = "Kestrel" if "kestrel" in triggers else rng.choice(TIERS)
    project = "Tamarind" if "tamarind" in triggers else rng.choice(PROJECTS)
    text = temp.format(p=product) + f" This is for Project {project}."
    state = (f"Ticket #{rng.randint(10000, 99999)} from {rng.choice(T.NAMES)} at {rng.choice(T.COMPANIES)} "
             f"(customer tier: {tier}):\n\"{text}\"")
    gold = {"team": "infrastructure" if "helios" in triggers else "sales" if "orchard" in triggers else default}
    if "kestrel" in triggers: gold["urgency"] = 3
    if "tamarind" in triggers: gold["legal"] = 1
    return dict(state=state, kind=kind, triggers=tuple(triggers), gold=gold)


def build(seed, split, combos=False):
    rng = random.Random(seed * 7 + (split == "test"))
    out = []
    for kind, temps, default in (("billing", T.BILLING, "billing"), ("sales", T.SALES, "sales"), ("infra", T.INFRA, "infrastructure")):
        half = len(temps) // 2
        for t in (temps[:half] if split == "train" else temps[half:]):
            sets = [()] + [(r,) for r in RULES]
            if combos:
                sets += [("helios", "kestrel"), ("orchard", "tamarind")]
            for trig in sets:
                out.append(_item(rng, kind, t, default, trig))
    return out


def with_policy(state):
    return POLICY + "\n\n" + state


def affected(item, q):
    """Does a rule present in the item govern question q?"""
    return any(RULE_Q[r] == q for r in item["triggers"])
