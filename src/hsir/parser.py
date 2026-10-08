"""Deterministic rule-based recipe parser: text -> semantic instruction stream.

Conservative by design: any sentence that does not match the grammar exactly
becomes a `U` (unsupported) instruction and its bytes travel in the residual.
The parser never consults ingredient lists, titles or other metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import lexicon as L
from .isa import Instr
from .state import State, transition

_SENT_RE = re.compile(r"[^.!?\n]+[.!?]*")
_TOK_RE = re.compile(
    r"[0-9]+(?:[./-][0-9]+)*"        # numbers, fractions, ranges
    r"|°[fc]?"                 # degree sign
    r"|[a-z]+"                 # words (ASCII only)
    r"|,"                      # comma
    r"|[.!?]"                  # terminal punctuation
    r"|\S"                     # anything else -> forces unsupported
)
_SKIP_MODS = {"about", "approximately", "approx", "another", "more"}
ZERO_ARG_OK = {"SERVE", "WAIT", "COOL", "SIMMER", "BOIL", "COOK", "BAKE",
               "STIR", "HEAT", "DRAIN", "COVER", "MIX", "WHISK",
               "SEASON"}


class Unsupported(Exception):
    pass


@dataclass
class SentenceStat:
    supported: bool
    nbytes: int
    reason: str = ""


@dataclass
class ParseResult:
    instrs: list[Instr]
    sentences: list[SentenceStat]
    n_entities: int = 0
    n_versions: int = 0
    n_anaphora: int = 0
    # one dict per supported sentence, in order: observed surface choices
    surface: list = field(default_factory=list)

    @property
    def any_supported(self) -> bool:
        return any(s.supported for s in self.sentences)


def tokenize(sentence: str) -> list[str]:
    return _TOK_RE.findall(sentence.lower())


def _isnum(t: str | None) -> bool:
    return bool(t) and t[0] in "0123456789"


def _is_word(t: str) -> bool:
    return t.isalpha() and t.isascii()


class _Cursor:
    def __init__(self, toks: list[str]):
        self.t = toks
        self.i = 0

    def peek(self, k: int = 0) -> str | None:
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def take(self) -> str:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def done(self) -> bool:
        return self.i >= len(self.t)


def _parse_np(c: _Cursor):
    """Return (qty, lexeme, is_anaphor, determiner-or-None) or None."""
    qty = None
    save = c.i
    p = c.peek()
    if p is not None and _isnum(p):
        num = c.take()
        unit = c.peek()
        if unit in L.QTY_UNITS:
            c.take()
            qty = (num, unit)
        else:
            qty = (num, "-")  # bare count: "2 eggs"
    det = None
    if c.peek() in L.DETERMINERS:
        det = c.take()
    headed = qty is not None or c.i > save  # determiner or quantity seen
    words = []
    while len(words) < 3:
        p = c.peek()
        if (p is None or not _is_word(p)
                or p in L.TEMP_UNITS or p in L.DUR_UNITS or p in _SKIP_MODS):
            break
        if p in L.NP_STOP:
            # A verb-form word may serve as a modifier ("brown sugar") when the
            # NP is headed by a determiner/quantity or another NP word follows.
            nxt = c.peek(1)
            as_modifier = (p in L.VERBS and p not in L.NP_STOP_HARD and (
                headed or words or (nxt is not None and _is_word(nxt)
                                    and nxt not in L.NP_STOP)))
            if not as_modifier:
                break
        words.append(c.take())
    if words and words[-1] in L.VERBS and words[-1] in L.NP_STOP and not headed:
        # never end an unheaded NP on a verb form ("... and stir")
        c.i -= 1
        words.pop()
    if not words:
        c.i = save
        return None
    lex = "_".join(words)
    return qty, lex, (len(words) == 1 and words[0] in L.ANAPHORS), det


def _parse_literal(c: _Cursor, units: set[str]):
    while c.peek() in _SKIP_MODS:
        c.take()
    p = c.peek()
    if p is None or not _isnum(p):
        raise Unsupported("expected number")
    num = c.take()
    us = []
    while c.peek() in units:
        us.append(c.take())
    if not us:
        raise Unsupported("expected unit")
    return num, "_".join(u.replace("°", "deg") for u in us)


def _np_obs(np) -> dict:
    qty, lex, ana, det = np
    return {"det": "" if det is None else det,
            "ref": lex if ana and lex in L.PRONOUNS else "np"}


def _parse_sentence(toks: list[str], st: State):
    # strip terminal punctuation
    while toks and toks[-1] in (".", "!", "?"):
        toks = toks[:-1]
    c = _Cursor(toks)
    led = False
    while c.peek() in L.LEADERS or c.peek() == "and":
        c.take()
        led = True
    v = c.peek()
    if v not in L.VERBS:
        raise Unsupported("no leading verb")
    c.take()
    op = L.VERBS[v]

    nps = []
    conj = []                      # separator consumed after each NP
    while True:
        np = _parse_np(c)
        if np is None:
            break
        nps.append(np)
        if c.peek() == ",":
            c.take()
            if c.peek() == "and":
                c.take()
                conj.append("oxford")
            else:
                conj.append("comma")
            continue
        if c.peek() == "and":
            c.take()
            conj.append("plain")
            continue
        break
    if nps and c.i > 0 and c.t[c.i - 1] in (",", "and"):
        raise Unsupported("dangling conjunction")

    dest = temp = dur = None
    while not c.done():
        p = c.peek()
        if p in _SKIP_MODS:
            c.take()
            continue
        if p in ("at", "to") and c.peek(1) is not None and (
                _isnum(c.peek(1)) or c.peek(1) in _SKIP_MODS):
            if temp:
                raise Unsupported("duplicate temperature")
            c.take()
            temp = _parse_literal(c, L.TEMP_UNITS)
            continue
        if p == "for":
            if dur:
                raise Unsupported("duplicate duration")
            c.take()
            dur = _parse_literal(c, L.DUR_UNITS)
            continue
        if p in L.PREPOSITIONS and dest is None:
            c.take()
            np = _parse_np(c)
            if np is None or np[0] is not None:
                raise Unsupported("bad destination")
            dest = (p, np)
            continue
        raise Unsupported(f"trailing token {p!r}")

    if not nps and op not in ZERO_ARG_OK:
        raise Unsupported("missing object")

    # ---- entity resolution (mutates a scratch copy; committed on success)
    work = st.copy()
    out: list[Instr] = []
    anaph = 0

    def resolve(lex: str, is_anaphor: bool) -> int:
        nonlocal anaph
        if is_anaphor:
            if work.focus is not None:
                anaph += 1
                return work.focus
            # only pronouns fall back to the last-touched entity; a lexical
            # anaphor ("the mixture") with no composite is a new entity
            if lex in L.PRONOUNS and work.last is not None:
                anaph += 1
                return work.last
        if lex in work.by_lexeme and not is_anaphor:
            return work.by_lexeme[lex]
        eid = work.new(lex)
        out.append(Instr("ENT", (f"e{eid}", lex)))
        return eid

    arg_ids = []
    qtys = []
    for qty, lex, ana, _det in nps:
        eid = resolve(lex, ana)
        arg_ids.append(eid)
        if qty:
            qtys.append((eid, qty))
    dest_id = None
    if dest:
        _, (_, dlex, dana, _) = dest
        dest_id = resolve(dlex, dana)

    for eid, (num, unit) in qtys:
        out.append(Instr("QTY", (work.ref(eid), num, unit)))
    out.append(Instr(op, tuple(work.ref(e) for e in arg_ids)))
    if dest:
        out.append(Instr("DEST", (dest[0], work.ref(dest_id))))
    if temp:
        out.append(Instr("TEMP", temp))
    if dur:
        out.append(Instr("DUR", dur))

    # ---- state transitions (shared with every decoder)
    out.extend(transition(work, op, arg_ids, dest_id))

    # commit
    st.entities, st.by_lexeme, st.focus, st.last = (
        work.entities, work.by_lexeme, work.focus, work.last)
    obs = {"verb": v, "led": led,
           "nps": [_np_obs(n) for n in nps],
           "dest": _np_obs(dest[1]) if dest else None}
    if len(nps) >= 3:
        last = conj[len(nps) - 2] if len(conj) >= len(nps) - 1 else None
        early_ok = all(x == "comma" for x in conj[:len(nps) - 2])
        obs["list"] = last if last in ("oxford", "plain") and early_ok else None
    return out, anaph, obs


def parse(text: str) -> ParseResult:
    st = State()
    instrs: list[Instr] = []
    stats: list[SentenceStat] = []
    surface: list[dict] = []
    anaph = 0
    prev_end = None          # trimmed end offset of the previous sentence
    prev_supported = False
    for m in _SENT_RE.finditer(text):
        sent = m.group(0)
        if not sent.strip():
            continue
        start = m.start() + (len(sent) - len(sent.lstrip()))
        end = m.end() - (len(sent) - len(sent.rstrip()))
        nbytes = len(sent.encode("utf-8"))
        try:
            body, a, obs = _parse_sentence(tokenize(sent), st)
        except Unsupported as e:
            instrs.append(Instr("U"))
            stats.append(SentenceStat(False, nbytes, str(e)))
            prev_end, prev_supported = end, False
            continue
        anaph += a
        instrs.append(Instr("S"))
        instrs.extend(body)
        stats.append(SentenceStat(True, nbytes))
        # observed surface facts (consumed only by surface.encode)
        first = text[start]
        obs["case"] = (None if obs.pop("led") or not first.isalpha()
                       else ("upper" if first.isupper() else "lower"))
        gap = text[prev_end:start] if prev_end is not None else None
        obs["sep_before"] = (gap if prev_supported and gap is not None
                             and gap.strip() == "" else None)
        surface.append(obs)
        prev_end, prev_supported = end, True
    nver = sum(e.ver for e in st.entities)
    return ParseResult(instrs, stats, len(st.entities), nver, anaph, surface)
