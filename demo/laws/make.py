# SPDX-License-Identifier: AGPL-3.0-only
"""Write demo/laws/charter.md, the charter the village demo reads as a knowledge pack.

    python3 demo/laws/make.py

Title I (rules of conduct) is hand-written and is what changes behaviour. The later titles are the
ordinary bulk of a town code (guilds, fees, fines), composed from a clause bank with a fixed seed.
"""
import random, os
OUT = os.path.dirname(os.path.abspath(__file__))
TRADES = ["bakers", "millers", "smiths", "farmers", "merchants", "fishers", "innkeepers", "carpenters", "weavers", "tanners", "masons", "herders", "brewers", "chandlers", "potters", "cartwrights"]
GOODS = ["bread", "flour", "nails", "barley", "cloth", "trout", "ale", "timber", "wool", "leather", "lime", "cheese", "candles", "pots", "salt", "honey"]
OFFICERS = ["the reeve", "the bailiff", "the town clerk", "the warden of the gate", "the market steward", "the sexton"]
def bulk(seed, theme, n_titles):
    r = random.Random(seed); out = []
    for t in range(n_titles):
        trade, good = TRADES[(t + seed) % len(TRADES)], GOODS[(t * 3 + seed) % len(GOODS)]
        out.append(f"\n## Title {['II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX','XXI','XXII','XXIII','XXIV','XXV'][t]}. Of the {trade} ({theme})\n")
        arts = [
            f"The {trade} of Emberfall form one guild. The guild keeps a roll of its masters, journeymen and apprentices, and {r.choice(OFFICERS)} may read the roll on any working day.",
            f"No one may sell {good} within the walls unless entered on the roll, on pain of a fine of {r.randint(2, 9)} coins for the first offence and {r.randint(10, 24)} coins for each after.",
            f"An apprentice to the {trade} serves {r.randint(3, 7)} years. The master feeds and houses the apprentice and may not put the apprentice to work before dawn or after the evening bell.",
            f"The measure for {good} is kept at the market cross. A seller found with a short measure forfeits the goods on the stall and pays {r.randint(3, 8)} coins to the common chest.",
            f"The guild pays {r.randint(12, 40)} coins to the common chest at each quarter day, and in return holds its pitch at the market free of stallage.",
            f"Disputes between two {trade} are heard first by the guild's two eldest masters; an appeal lies to {r.choice(OFFICERS)} within {r.randint(5, 14)} days.",
            f"A master who takes another master's apprentice, customer under contract, or tools pays {r.randint(6, 18)} coins and restores what was taken.",
            f"Waste from the trade is carried beyond the south ditch. Fouling the lake, the mill race or the well costs {r.randint(4, 12)} coins and the labour of clearing it.",
            f"The widow or widower of a master may keep the shop for {r.randint(1, 3)} years with a journeyman, after which the shop passes to an heir on the roll or is sold by the guild.",
            f"On the death of a master with no heir, {r.choice(OFFICERS)} seals the shop, lists its stock before two witnesses, and pays debts in the order: wages, the common chest, other creditors.",
            f"Strangers bringing {good} to market pay a toll of one coin in every {r.randint(12, 30)} and may sell only from the third bell to the sixth.",
            f"The price of {good} in a year of dearth is set by {r.choice(OFFICERS)} with two masters of the guild, and selling above it costs treble the excess.",
        ]
        r.shuffle(arts)
        for k, a in enumerate(arts[: r.randint(9, 12)], 1): out.append(f"Article {t + 2}.{k}. {a}")
    return "\n".join(out)

HEAD = "The actions named in this Charter are the ones a villager may choose: work, eat, sleep, chat, hide, look, help, flee, party."
BOOKS = {
"charter": ("The Charter of Emberfall", "the common peace", 3, f"""# The Charter of Emberfall

Granted to the free village of Emberfall. This Charter is the whole law of the village and binds every villager, of every age and trade. The morning bell rings at seven and the evening bell at six. {HEAD}

## Title I. Rules of conduct

Article 1.1. Work. From the morning bell to the evening bell, every adult of working age is at work at their own trade.
Article 1.2. School. From the morning bell to the evening bell, every child is in school: a schoolchild's work is school. No child leaves class for news, for a crowd, or for a stranger.
Article 1.3. Elders. Elders are excused from work. By day they keep the square, where they chat and hear the village's disputes.
Article 1.4. The tavern. By day the tavern serves a meal only to a villager who is starving. No one else eats or idles there before the evening bell.
Article 1.5. Strangers. When the crier announces a stranger at the gate, only the elders go to look and greet them. Everyone else carries on exactly as before.
Article 1.6. Fire. When the crier calls a fire, every adult of working age goes and helps. Children stay in school and elders stay in the square.
Article 1.7. Wolves and raiders. When the crier calls wolves or raiders, every villager hides: they lock themselves in at home.
Article 1.8. Evening. After the evening bell the village is free: villagers eat, meet friends, celebrate or go home to bed as they wish.
"""),
}
for name, (title, theme, seed, head) in BOOKS.items():
    text = head + bulk(seed, theme, 22)
    open(os.path.join(OUT, name + ".md"), "w").write(text.strip() + "\n")
    print(name, len(text.split()), "words")
