"""Synthetic support tickets for a private routing rule d3-ultra cannot know.

Rule (Acme): every ticket that mentions Helios goes to Infrastructure, whatever it is about.
Cells: Helios x {billing, sales} scenario -> infrastructure (the rule flips the default);
control product x {billing, sales} -> the default team; {Helios, control} x infra -> infrastructure.
Templates are split into train and test halves, so a test item never shares wording with a training item.
"""
import random

POLICY = ("Acme routing policy: Helios is run by the Infrastructure team end to end, including its billing "
          "and its sales. Every ticket that mentions Helios goes to Infrastructure, whatever it is about.")

TEAM_Q = {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": {"billing": "Billing", "infrastructure": "Infrastructure", "sales": "Sales"}}}
TEAMS = list(TEAM_Q["team"]["criteria"])

BILLING = [
    "Hi, I was charged twice this month for our {p} subscription. Can you refund the duplicate charge?",
    "Our invoice for {p} shows the wrong VAT number. Please reissue it with the correct one.",
    "The card on file for {p} was declined at renewal and now the account says past due. How do I pay?",
    "I need a copy of last quarter's receipts for {p} for our accountants.",
    "We cancelled {p} in March but you still billed us in April. Please refund it.",
    "Can you change the billing address printed on our {p} invoices? We moved offices.",
    "Why did our {p} bill go up by 40% this cycle? Nothing changed on our side.",
    "Please switch our {p} payment method from credit card to bank transfer.",
    "There is a charge on my statement labelled {p} that I don't recognise. What is it for?",
    "Our {p} invoice was sent to an employee who left. Please send future invoices to finance@ourco.example.",
    "We were promised a credit for the downtime on {p} but it never showed up on the bill.",
    "I paid the {p} invoice twice by mistake, once by card and once by wire. Can you return one?",
]
SALES = [
    "We'd like a quote for 200 seats of {p} for our engineering org.",
    "Is there enterprise pricing for {p}? We're comparing vendors this month.",
    "Could someone give our team a demo of {p} next week? We're thinking of buying.",
    "Do you offer volume discounts on {p} if we commit to three years?",
    "We're a reseller and want to bundle {p} with our services. Who do we talk to about a partnership?",
    "Our {p} trial ends Friday; we want to buy but need a bit more time to get budget approval.",
    "What would it cost to upgrade from the starter plan of {p} to the business plan for 50 users?",
    "We're expanding to Europe and want to add 120 more {p} licenses. Can you send a proposal?",
    "Can we get nonprofit pricing on {p}? We're a registered charity.",
    "I'd like to talk to an account manager about rolling {p} out across all our offices.",
    "Does {p} come with an annual prepay discount? Our CFO wants numbers before renewal.",
    "We want to buy {p} for a new subsidiary. Can it go on a separate contract?",
]
INFRA = [
    "Our {p} dashboard has been timing out for the last hour. Is there an outage?",
    "API calls to {p} are returning 500 errors since this morning's deploy.",
    "Latency on {p} jumped from 50 ms to 2 s in the eu-west region.",
    "Nightly backups for our {p} instance have been failing for three days.",
    "The TLS certificate on our {p} endpoint expired and clients refuse to connect.",
    "We need to migrate our {p} workloads to a different region with minimal downtime.",
    "Our {p} cluster keeps running out of disk and the autoscaler isn't kicking in.",
    "Webhooks from {p} stopped arriving at our servers around 3 am UTC.",
]
CONTROLS = ["Nimbus", "Orion", "Vega", "Lumen", "Cobalt", "Quasar", "Tidal", "Beacon"]
NAMES = ["Mina", "Rahul", "Sofia", "Kenji", "Amara", "Lukas", "Priya", "Diego", "Hana", "Omar", "Elena", "Tariq"]
COMPANIES = ["Northwind", "Brightpath", "Kelvin Labs", "Marrow & Co", "Bluefin", "Stackhouse", "Quill", "Ferro"]


def _ticket(rng, text):
    return (f"Ticket #{rng.randint(10000, 99999)} from {rng.choice(NAMES)} at {rng.choice(COMPANIES)}:\n"
            f"\"{text}\"")


def build(seed=0):
    """{split: [item]}; item = dict(state, kind, product, gold, rule)."""
    rng = random.Random(seed)
    out = {"train": [], "test": []}
    for kind, temps, default in (("billing", BILLING, "billing"), ("sales", SALES, "sales"), ("infra", INFRA, "infrastructure")):
        half = len(temps) // 2
        for split, part in (("train", temps[:half]), ("test", temps[half:])):
            for t in part:
                for product in ["Helios", rng.choice(CONTROLS)]:
                    helios = product == "Helios"
                    out[split].append(dict(
                        state=_ticket(rng, t.format(p=product)), kind=kind, product=product,
                        gold="infrastructure" if helios else default, rule=helios and kind != "infra"))
    return out


def with_policy(state):
    return POLICY + "\n\n" + state


def build_train(seeds=(0, 1, 2)):
    """The train templates with fresh names, numbers and control products per seed (one Helios + one control each)."""
    out = []
    for s in seeds:
        out += build(s)["train"]
    return out
