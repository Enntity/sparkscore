"""A long support handbook (~3K tokens) with seven override rules spread through ordinary content, and tickets for it.

Override rules (position in the handbook): wisteria (legal, early), helios (team, early), kestrel (urgency, middle),
lattice (team, middle), osprey (urgency, late-middle), tamarind (legal, late), orchard (team, late).
Control products, tiers and projects are described in the handbook too, without overrides.
"""
import random
import tickets as T
import rules4 as R4

RULE_Q = {"helios": "team", "orchard": "team", "lattice": "team", "kestrel": "urgency", "osprey": "urgency",
          "tamarind": "legal", "wisteria": "legal"}
RULES = list(RULE_Q)
GOLD = {"helios": ("team", "infrastructure"), "orchard": ("team", "sales"), "lattice": ("team", "billing"),
        "kestrel": ("urgency", 3), "osprey": ("urgency", 0), "tamarind": ("legal", 1), "wisteria": ("legal", 1)}
QS = R4.QS
PRODUCTS = ["Nimbus", "Vega", "Lumen", "Cobalt", "Quasar", "Tidal", "Beacon"]
TIERS = ["Sparrow", "Finch", "Wren", "Heron"]
PROJECTS = ["Juniper", "Bramble", "Saffron", "Marigold"]

_CATALOG = {
    "Nimbus": "our hosted file-sync service for small teams; outages page the on-call engineer through the status bot",
    "Vega": "the analytics dashboard bundled with business plans; feature requests go to the product board",
    "Lumen": "a lightweight uptime monitor sold per check; its invoices are issued monthly in arrears",
    "Cobalt": "the self-hosted deployment of our core platform, licensed annually per node",
    "Quasar": "the realtime messaging API, metered per million messages",
    "Tidal": "a data-export connector that syncs records to customer warehouses every hour",
    "Beacon": "the mobile push service, priced by monthly active devices",
    "Helios": "our managed database platform",
    "Orchard": "the partner marketplace where resellers list bundles",
    "Lattice": "the usage-metering add-on that sits on top of every plan",
}

POLICY = "\n\n".join([
    "ACME CUSTOMER SUPPORT HANDBOOK (internal, version 14)",
    "1. Purpose. This handbook tells support agents how to triage incoming tickets: which team owns a ticket, how "
    "urgent it is, and whether it needs review by the legal team before anyone replies. Where a section below states "
    "an override, the override wins over the general guidance. Read the whole handbook before your first shift.",
    "2. Tone and replies. Greet the customer by name, restate the problem in one sentence, and say what happens next. "
    "Do not promise dates you do not control. Avoid internal jargon such as ticket states or team nicknames. Sign "
    "replies with your first name and the team name. Use the macros in section 11 for common cases.",
    "3. Legal review. Tickets that involve subpoenas, law-enforcement requests, or threats of litigation always go "
    "to legal review. Ordinary contract questions, invoices and refund requests do not need legal review unless an "
    "override applies. Override: Project Wisteria is the codename of an acquisition that has not been announced. "
    "Any ticket that mentions Project Wisteria needs legal review, whatever it is about, and must not be answered "
    "until legal has cleared it.",
    "4. Product catalog and ownership. By default, billing questions go to Billing, purchasing and pricing questions "
    "go to Sales, and outages, errors and performance problems go to Infrastructure. " + " ".join(
        f"{p}: {d}." for p, d in _CATALOG.items() if p not in ("Helios", "Orchard", "Lattice"))
    + " Helios: " + _CATALOG["Helios"] + ". Override: Helios is run by the Infrastructure team end to end, including "
    "its billing and its sales. Every ticket about Helios goes to Infrastructure, whatever it is about.",
    "5. Working hours and holidays. The support desk is staffed from 07:00 to 22:00 UTC on weekdays and from 09:00 to "
    "17:00 UTC at weekends. On public holidays in Ireland the desk runs a reduced rota; check the rota calendar before "
    "promising a callback. Agents swap shifts through the rota tool, never by private arrangement.",
    "6. Customer tiers and response targets. Sparrow is the free tier, with community support only and no response "
    "target. Finch customers get a reply within two business days. Wren customers get a reply within one business "
    "day. Heron customers get a reply within eight business hours and a named account contact. Override: Kestrel-tier "
    "customers have a contractual one-hour response. Any ticket from a Kestrel-tier customer is Right now urgent, "
    "whatever it is about.",
    "7. Billing operations. Invoices are issued on the first business day of the month. Refunds for duplicate charges "
    "are approved by Billing without escalation up to the value of one month's subscription. Changes of billing "
    "address, VAT number or payment method are handled by Billing. Lattice: " + _CATALOG["Lattice"] + ". Override: "
    "everything about Lattice is handled by the Billing team, including its outages and its sales. Every ticket about "
    "Lattice goes to Billing, whatever it is about.",
    "8. Security incidents. If a customer reports a suspected breach, leaked credentials or unexpected account "
    "access, open a security incident from the ticket and tell the customer that the security team will contact "
    "them. Never ask the customer to send passwords or keys in a ticket.",
    "9. Urgency guidance. Production outages and data loss are Right now. Degraded performance with a workaround is "
    "Today. Billing errors and purchasing questions are usually This week, and general questions Can wait. Override: "
    "Osprey is our best-effort community tier for open-source projects. Any ticket from an Osprey-tier customer can "
    "wait, whatever it is about, including outages; reply when the queue allows.",
    "10. Projects and codenames. Customer projects named in tickets are usually harmless labels. Project Juniper, "
    "Project Bramble, Project Saffron and Project Marigold are ordinary customer onboarding programmes and need no "
    "special handling. Override: Project Tamarind is under a confidentiality order. Any ticket that mentions Project "
    "Tamarind needs legal review, whatever it is about.",
    "11. Macros. Use 'refund-duplicate' for duplicate charges, 'invoice-reissue' for invoice corrections, "
    "'quote-request' for pricing questions, 'outage-ack' for outage reports and 'feature-thanks' for feature "
    "requests. Edit every macro before sending; never send one unchanged.",
    "12. Partners and resellers. Resellers buy at partner prices and bill their own customers. Questions from "
    "resellers about margins or co-marketing go to the partner manager in Sales. Orchard: " + _CATALOG["Orchard"]
    + ". Override: Orchard is sold and supported by the Sales team. Every ticket about Orchard goes to Sales, "
    "whatever it is about, including its outages and its invoices.",
    "13. Closing tickets. Close a ticket only when the customer confirms the fix or after seven days without a reply "
    "following a proposed solution. Tag every closed ticket with the product and the root cause.",
])


def _item(rng, kind, temp, default, trig):
    product = {"helios": "Helios", "orchard": "Orchard", "lattice": "Lattice"}.get(trig) or rng.choice(PRODUCTS)
    tier = {"kestrel": "Kestrel", "osprey": "Osprey"}.get(trig) or rng.choice(TIERS)
    project = {"tamarind": "Tamarind", "wisteria": "Wisteria"}.get(trig) or rng.choice(PROJECTS)
    text = temp.format(p=product) + f" This is for Project {project}."
    state = (f"Ticket #{rng.randint(10000, 99999)} from {rng.choice(T.NAMES)} at {rng.choice(T.COMPANIES)} "
             f"(customer tier: {tier}):\n\"{text}\"")
    gold = {"team": default}
    if trig:
        q, g = GOLD[trig]; gold[q] = g
    return dict(state=state, kind=kind, triggers=(trig,) if trig else (), gold=gold)


def build_test(seed=0, per=5):
    """`per` items for no trigger and for each rule, on the test templates."""
    rng = random.Random(seed)
    temps = []
    for kind, ts, default in (("billing", T.BILLING, "billing"), ("sales", T.SALES, "sales"), ("infra", T.INFRA, "infrastructure")):
        temps += [(kind, t, default) for t in ts[len(ts) // 2:]]
    out = []
    for trig in [None] + RULES:
        for kind, t, default in rng.sample(temps, per):
            out.append(_item(rng, kind, t, default, trig))
    return out
