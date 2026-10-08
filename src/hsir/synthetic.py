"""Synthetic recipe-like documents. FOR UNIT TESTS ONLY.

Because the generator and the parser share vocabulary, synthetic corpora are
circular for the feasibility question: any Phase II report built from them
is marked synthetic and its decision is forced to PENDING_REAL_DATA.
"""
from __future__ import annotations

import random

ING = ["flour", "sugar", "butter", "eggs", "milk", "salt", "pepper", "onions",
       "garlic", "chicken", "rice", "carrots", "cream cheese", "brown sugar",
       "vanilla", "potatoes", "tomatoes", "olive oil", "baking soda", "cheese"]
TOOLS = ["bowl", "large bowl", "pan", "skillet", "baking dish", "pot",
         "saucepan", "sheet"]
TRANSFORM = ["Chop", "Dice", "Slice", "Grate", "Peel", "Season", "Drain"]
COOK = ["Bake", "Simmer", "Cook", "Fry", "Boil", "Heat", "Cool"]
UNSUPPORTED = [
    "Cook until golden brown, stirring occasionally.",
    "If the batter is too thick, add a little more milk.",
    "Do not overmix!",
    "Enjoy with friends (serves 4).",
    "Mélangez bien et laissez reposer 10 min.",
    "Bring to a boil; reduce heat.",
    "Let stand 5 minutes before cutting.",
    "Remove from heat when done.",
]


def _np(rng: random.Random, ing: str) -> str:
    r = rng.random()
    if r < 0.15:
        return f"{rng.randint(1, 4)} cups {ing}"
    if r < 0.6:
        return f"the {ing}"
    return ing


def doc(rng: random.Random) -> str:
    sents = []
    if rng.random() < 0.6:
        sents.append(f"Preheat oven to {rng.choice([325, 350, 375, 400, 425])} "
                     f"{rng.choice(['degrees F', 'degrees', '°F', 'degrees C'])}.")
    for _ in range(rng.randint(2, 7)):
        k = rng.random()
        if k < 0.25:
            ings = rng.sample(ING, rng.randint(2, 4))
            nps = [_np(rng, i) for i in ings]
            body = (", ".join(nps[:-1]) + (", and " if rng.random() < 0.5 else " and ")
                    + nps[-1])
            s = f"{rng.choice(['Mix', 'Combine', 'Whisk'])} {body}"
            if rng.random() < 0.5:
                s += f" in a {rng.choice(TOOLS)}"
            sents.append(s + ".")
        elif k < 0.45:
            s = f"{rng.choice(TRANSFORM)} {_np(rng, rng.choice(ING))}"
            sents.append(s + ".")
        elif k < 0.6:
            s = (f"Add {_np(rng, rng.choice(ING))} to "
                 f"{rng.choice(['the mixture', 'it', 'the ' + rng.choice(TOOLS)])}")
            sents.append(s + ".")
        elif k < 0.75:
            obj = rng.choice(["", " it", " the mixture", " " + _np(rng, rng.choice(ING))])
            s = f"{rng.choice(COOK)}{obj}"
            if rng.random() < 0.5:
                s += f" for {rng.choice(['about ', ''])}{rng.choice(['5', '10', '20-25', '45'])} minutes"
            if rng.random() < 0.2:
                s += f" at {rng.choice([350, 400])} degrees"
            sents.append(s + ".")
        elif k < 0.85:
            sents.append(f"{rng.choice(['Then', 'Next', 'Gently'])} stir "
                         f"{rng.choice(['it', 'the mixture'])}.")
        else:
            sents.append(rng.choice(UNSUPPORTED))
    sep = rng.choice(["\n", " ", "\n"])
    return sep.join(sents)


def corpus(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [doc(rng) for _ in range(n)]
