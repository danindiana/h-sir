"""Frozen rule tables for the recipe ISA.

These tables are hand-written and corpus-independent. They are part of the
shared artifact cost L_shared and are hashed into every Phase II manifest.
Any change must be logged in RULES_CHANGELOG.md and bumps RULES_VERSION.
"""

RULES_VERSION = "0.3.2"

# surface verb -> operator.  The first surface form listed for an operator is
# the one the realizer emits.
VERBS: dict[str, str] = {
    "preheat": "PREHEAT",
    "chop": "CHOP", "dice": "CHOP", "mince": "CHOP",
    "slice": "SLICE", "cut": "SLICE",
    "grate": "GRATE", "shred": "GRATE",
    "peel": "PEEL",
    "mix": "MIX", "combine": "MIX", "blend": "MIX",
    "stir": "STIR", "fold": "STIR",
    "whisk": "WHISK", "beat": "WHISK",
    "add": "ADD",
    "pour": "POUR",
    "place": "PLACE", "put": "PLACE", "transfer": "PLACE",
    "spread": "SPREAD",
    "sprinkle": "SPRINKLE", "top": "SPRINKLE",
    "season": "SEASON",
    "heat": "HEAT", "warm": "HEAT", "melt": "HEAT",
    "boil": "BOIL", "bring": "BOIL",
    "simmer": "SIMMER",
    "cook": "COOK",
    "fry": "FRY", "saute": "FRY", "brown": "FRY",
    "bake": "BAKE", "roast": "BAKE",
    "cool": "COOL", "chill": "COOL", "refrigerate": "COOL",
    "drain": "DRAIN",
    "cover": "COVER",
    "serve": "SERVE",
    "let": "WAIT",
}

# Operators whose arguments are merged into one new composite entity.
COMBINING = {"MIX", "WHISK"}
# Operators that move arg into a destination; destination absorbs arg.
ABSORBING = {"ADD", "POUR", "PLACE", "SPREAD", "SPRINKLE"}
# Operators that change the state of each argument in place.
TRANSFORMING = {"CHOP", "SLICE", "GRATE", "PEEL", "STIR", "SEASON", "HEAT",
                "BOIL", "SIMMER", "COOK", "FRY", "BAKE", "COOL", "DRAIN",
                "COVER", "WAIT"}
# Operators that observe but do not transform.
NONTRANSFORMING = {"PREHEAT", "SERVE"}

DETERMINERS = {"the", "a", "an", "some", "your"}
PREPOSITIONS = {"in", "into", "to", "on", "onto", "over", "with", "from"}
CONJUNCTIONS = {"and", ","}
# Leading discourse adverbs the parser tolerates (they go to the residual).
LEADERS = {"then", "next", "now", "finally", "gently", "slowly", "carefully"}

ANAPHORS = {"it", "them", "mixture", "everything", "all"}
# Anaphors that may fall back to the last-touched entity when no composite exists.
PRONOUNS = {"it", "them", "everything", "all"}

QTY_UNITS = {"cup", "cups", "c", "tbsp", "tablespoon", "tablespoons", "tsp",
             "teaspoon", "teaspoons", "oz", "ounce", "ounces", "lb", "lbs",
             "pound", "pounds", "g", "grams", "ml", "pinch", "dash", "can",
             "cans", "package", "pkg", "stick", "sticks", "clove", "cloves"}
TEMP_UNITS = {"degrees", "degree", "f", "c", "fahrenheit", "celsius", "°",
              "°f", "°c"}
DUR_UNITS = {"minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs",
             "second", "seconds", "sec", "secs"}

# Words that may never be part of a noun phrase.
NP_STOP = (set(VERBS) | DETERMINERS | PREPOSITIONS | {"and", "or", "for",
           "at", "until", "about", "then", "if", "when", "while", "until",
           "not", "well", "until", "each", "per", "of"})

# Non-verb stop words: never part of a noun phrase.
NP_STOP_HARD = NP_STOP - set(VERBS)

COMPOSITE_LEXEME = "mixture"


def realize_verb(op: str) -> str:
    for surface, o in VERBS.items():
        if o == op:
            return surface
    raise KeyError(op)


def table_bytes() -> bytes:
    """Canonical serialization of every rule table, for hashing and MDL."""
    import json
    tables = {
        "RULES_VERSION": RULES_VERSION,
        "VERBS": VERBS,
        "COMBINING": sorted(COMBINING),
        "ABSORBING": sorted(ABSORBING),
        "TRANSFORMING": sorted(TRANSFORMING),
        "NONTRANSFORMING": sorted(NONTRANSFORMING),
        "DETERMINERS": sorted(DETERMINERS),
        "PREPOSITIONS": sorted(PREPOSITIONS),
        "LEADERS": sorted(LEADERS),
        "ANAPHORS": sorted(ANAPHORS),
        "PRONOUNS": sorted(PRONOUNS),
        "QTY_UNITS": sorted(QTY_UNITS),
        "TEMP_UNITS": sorted(TEMP_UNITS),
        "DUR_UNITS": sorted(DUR_UNITS),
        "NP_STOP": sorted(NP_STOP),
        "NP_STOP_HARD": sorted(NP_STOP_HARD),
        "COMPOSITE_LEXEME": COMPOSITE_LEXEME,
    }
    return json.dumps(tables, sort_keys=True, separators=(",", ":")).encode()
