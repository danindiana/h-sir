"""Surface-factorized realization: y_hat = G(S, C).

C is a sparse sequence of categorical surface choices. Slots are enumerated
deterministically from S (and the enabled field set), so a choice is stored
as (gap to its slot, value id) and only when it differs from the default.
The byte-patch residual R = diff(G(S, C), x) still guarantees exactness, so
a wrong or missing choice can only cost bits, never correctness.

Schema (SURFACE_SCHEMA_VERSION, append-only value lists):
    SEP   separator before supported sentence j (j > 0)
    CASE  initial case of the verb
    VERB  lexical variant of the verb among its surface forms
    ART   article of an argument / destination NP without quantity
    LIST  serial comma in lists of >= 3 items
    REF   reference form of an argument / destination NP
"""
from __future__ import annotations

from . import lexicon as L
from .isa import Instr, STRUCTURAL

SURFACE_SCHEMA_VERSION = 1

SEP_VALUES = [" ", "\n", "", "\n\n", "  ", " \n", "\n ", "\r\n"]
CASE_VALUES = ["upper", "lower"]
ART_VALUES = ["the", "a", "an", "", "some", "your"]
LIST_VALUES = ["oxford", "plain"]
REF_VALUES = ["np", "it", "them", "everything", "all"]
FIELD_ORDER = ["SEP", "CASE", "VERB", "ART", "LIST", "REF"]

_GROUPS = {
    "C1": ("SEP", "LIST"),
    "C2": ("SEP", "LIST", "ART", "CASE"),
    "C3": ("SEP", "LIST", "ART", "CASE", "VERB"),
    "C4": ("SEP", "LIST", "ART", "CASE", "VERB", "REF"),
}
# name -> (enabled fields, permuted ids)
CSETS: dict[str, tuple[tuple[str, ...], bool]] = {
    "C0": ((), False),
    "C1": (_GROUPS["C1"], False),
    "C2": (_GROUPS["C2"], False),
    "C3": (_GROUPS["C3"], False),
    "C4": (_GROUPS["C4"], False),
    "C4P": (_GROUPS["C4"], True),   # fixed bijective permutation of value ids
}
CSET_IDS = list(CSETS)              # header byte = index; append-only


def verb_surfaces(op: str) -> list[str]:
    return [s for s, o in L.VERBS.items() if o == op]


def n_values(key: tuple, op: str) -> int:
    f = key[0]
    return {"SEP": len(SEP_VALUES), "CASE": len(CASE_VALUES),
            "ART": len(ART_VALUES), "LIST": len(LIST_VALUES),
            "REF": len(REF_VALUES)}.get(f) or len(verb_surfaces(op))


def _sentences(instrs: list[Instr]):
    cur = None
    for ins in instrs:
        if ins.op in ("S", "U"):
            if cur is not None:
                yield cur
            cur = [ins]
        else:
            if cur is None:
                raise ValueError("instruction before first sentence marker")
            cur.append(ins)
    if cur is not None:
        yield cur


def render(instrs: list[Instr], fields: tuple[str, ...], choose=None):
    """Realize text. `choose(key, nvals)` returns a value index for each
    enabled slot (default 0). Returns (text, [(key, op, nvals), ...])."""
    en = set(fields)
    slots: list[tuple] = []

    def pick(key, op):
        if key[0] not in en:
            return 0
        nv = n_values(key, op)
        if key[0] == "VERB" and nv < 2:
            return 0
        slots.append((key, op, nv))
        v = choose(key, nv) if choose else 0
        if not 0 <= v < nv:
            raise ValueError(f"choice out of range for {key}")
        return v

    lex: dict[str, str] = {}
    out: list[str] = []
    seps: list[str] = []
    j = -1
    for sent in _sentences(instrs):
        if sent[0].op == "U":
            if len(sent) != 1:
                raise ValueError("U sentence must be empty")
            continue
        j += 1
        qty: dict[str, tuple[str, str]] = {}
        action: Instr | None = None
        dest = temp = dur = None
        for ins in sent[1:]:
            if ins.op == "ENT":
                lex[ins.args[0]] = ins.args[1]
            elif ins.op == "QTY":
                qty[ins.args[0]] = (ins.args[1], ins.args[2])
            elif ins.op == "DEST":
                dest = ins.args
            elif ins.op == "TEMP":
                temp = ins.args
            elif ins.op == "DUR":
                dur = ins.args
            elif ins.op == "OUT":
                pass
            elif ins.op in STRUCTURAL:
                raise ValueError(f"unexpected {ins.op}")
            else:
                if action is not None:
                    raise ValueError("two actions in one sentence")
                action = ins
        if action is None:
            raise ValueError("supported sentence without action")
        op = action.op

        if j > 0:
            seps.append(SEP_VALUES[pick(("SEP", j), op)])
        case = CASE_VALUES[pick(("CASE", j), op)]
        verb = verb_surfaces(op)[pick(("VERB", j), op)]

        def np(ref: str, role) -> str:
            word = lex[ref.split(".")[0]].replace("_", " ")
            r = REF_VALUES[pick(("REF", j, role), op)]
            art_slot = ref not in qty
            a = ART_VALUES[pick(("ART", j, role), op)] if art_slot else None
            if r != "np":
                return r
            if not art_slot:
                n, u = qty[ref]
                return f"{n} {word}" if u == "-" else f"{n} {u} {word}"
            return f"{a} {word}" if a else word

        s = verb.capitalize() if case == "upper" else verb
        items = [np(r, i) for i, r in enumerate(action.args)]
        if items:
            if len(items) == 1:
                body = items[0]
            elif len(items) == 2:
                body = f"{items[0]} and {items[1]}"
            else:
                style = LIST_VALUES[pick(("LIST", j), op)]
                body = ", ".join(items[:-1]) + (
                    f", and {items[-1]}" if style == "oxford" else f" and {items[-1]}")
            s += " " + body
        if dest:
            s += f" {dest[0]} {np(dest[1], 'd')}"
        if temp:
            prep = "to" if op == "PREHEAT" else "at"
            s += f" {prep} {temp[0]} {temp[1].replace('_', ' ')}"
        if dur:
            s += f" for {dur[0]} {dur[1].replace('_', ' ')}"
        out.append(s + ".")
    text = out[0] if out else ""
    for sep, piece in zip(seps, out[1:]):
        text += sep + piece
    return text, slots


# ------------------------------------------------------------- observation
def _value_index(key: tuple, op: str, obs_val) -> int | None:
    f = key[0]
    table = {"SEP": SEP_VALUES, "CASE": CASE_VALUES, "ART": ART_VALUES,
             "LIST": LIST_VALUES, "REF": REF_VALUES}.get(f)
    if table is None:
        table = verb_surfaces(op)
    try:
        return table.index(obs_val)
    except ValueError:
        return None


def _observed(key: tuple, surface: list[dict]):
    """Observed source value for a slot, or the sentinel _UNOBSERVED."""
    f, j = key[0], key[1]
    o = surface[j]
    if f == "SEP":
        return o.get("sep_before", _UNOBSERVED)
    if f == "CASE":
        return o.get("case", _UNOBSERVED)
    if f == "VERB":
        return o.get("verb", _UNOBSERVED)
    if f == "LIST":
        return o.get("list", _UNOBSERVED)
    role = key[2]
    np = o["dest"] if role == "d" else o["nps"][role]
    if np is None:
        return _UNOBSERVED
    return np["det"] if f == "ART" else np["ref"]


_UNOBSERVED = object()


def _perm(v: int, n: int, permuted: bool) -> int:
    return (n - 1 - v) if permuted else v


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _rvarint(b: bytes, k: int) -> tuple[int, int]:
    n = shift = 0
    start = k
    while True:
        if k >= len(b):
            raise ValueError("truncated varint")
        c = b[k]
        k += 1
        n |= (c & 0x7F) << shift
        if not c & 0x80:
            if k - start > 1 and c == 0:
                raise ValueError("non-canonical varint")
            return n, k
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def encode(instrs: list[Instr], surface: list[dict] | None, cset: str):
    """Choose surface values from parser observations. Returns (C bytes, stats)."""
    fields, permuted = CSETS[cset]
    stats = {f: {"slots": 0, "emitted": 0, "unobserved": 0, "unrepresentable": 0}
             for f in fields}
    choices: dict[tuple, int] = {}

    def choose(key, nv):
        st = stats[key[0]]
        st["slots"] += 1
        if surface is None:
            st["unobserved"] += 1
            return 0
        ov = _observed(key, surface)
        if ov is _UNOBSERVED or ov is None:
            st["unobserved"] += 1
            return 0
        idx = _value_index(key, None if key[0] != "VERB" else _op_of(key), ov)
        if idx is None:
            st["unrepresentable"] += 1
            return 0
        if idx:
            st["emitted"] += 1
            choices[key] = idx
        return idx

    # VERB needs the op; resolve via a first pass that records slot ops
    _, slots = render(instrs, fields, None)
    ops = {k: op for k, op, _ in slots}

    def _op_of(key):
        return ops[key]

    _, slots = render(instrs, fields, choose)
    out = bytearray()
    prev = -1
    for idx, (key, op, nv) in enumerate(slots):
        v = choices.get(key, 0)
        if v:
            out += _varint(idx - prev - 1) + _varint(_perm(v, nv, permuted))
            prev = idx
    return bytes(out), stats


def decode(instrs: list[Instr], c: bytes, cset: str) -> dict[tuple, int]:
    fields, permuted = CSETS[cset]
    _, slots = render(instrs, fields, None)
    choices: dict[tuple, int] = {}
    k = 0
    prev = -1
    while k < len(c):
        gap, k = _rvarint(c, k)
        sid, k = _rvarint(c, k)
        idx = prev + 1 + gap
        if idx >= len(slots):
            raise ValueError("surface choice beyond last slot")
        key, op, nv = slots[idx]
        if sid >= nv:
            raise ValueError("surface value out of range")
        v = _perm(sid, nv, permuted)
        if v == 0:
            raise ValueError("default value must not be stored")
        choices[key] = v
        prev = idx
    return choices


def realize(instrs: list[Instr], cset: str = "C0", c: bytes = b"") -> str:
    fields, _ = CSETS[cset]
    if not fields:
        if c:
            raise ValueError("C0 carries no surface choices")
        return render(instrs, ())[0]
    choices = decode(instrs, c, cset)
    return render(instrs, fields, lambda key, nv: choices.get(key, 0))[0]


def symbols(instrs: list[Instr], c: bytes, cset: str) -> list[str]:
    """Model-facing symbols for C: a gap symbol and a field-scoped value."""
    fields, permuted = CSETS[cset]
    if not c:
        return []
    _, slots = render(instrs, fields, None)
    toks = []
    k = 0
    prev = -1
    while k < len(c):
        gap, k = _rvarint(c, k)
        sid, k = _rvarint(c, k)
        idx = prev + 1 + gap
        toks += [f"+{gap}", f"{slots[idx][0][0]}={sid}"]
        prev = idx
    return toks
