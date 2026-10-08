"""Serialization modes for the semantic stream.

Two independent axes:
    encoding  ascii   - space-separated tokens, one instruction per line
              compact - opcode byte + typed operands (varints, NUL-terminated
                        strings); no line terminators
    state     explicit - versioned refs, entity ids on ENT, composite ENT and
                         OUT lines serialized
              implicit - ids only; versions, composite ENT and OUT derived by
                         replaying state.transition in the decoder
              ckpt     - implicit + a CK snapshot of every entity's version
                         after every k-th action (verified on decode)

Named modes:
    S0    ascii   explicit     (v0.3.1 format, byte-identical)
    S3    compact explicit     S0 information, compact spelling
    S1    compact implicit
    S1a   ascii   implicit     (diagnostic: implicit state, verbose spelling)
    S2k4 / S2k8 / S2k16   compact ckpt, k = 4 / 8 / 16

Every decoder returns the canonical explicit instruction list and validates
it by replay, so an inconsistent stream raises instead of decoding silently.
"""
from __future__ import annotations

from . import lexicon as L
from .isa import Instr
from .state import State, transition

MODES: dict[str, tuple[str, str, int]] = {
    "S0": ("ascii", "explicit", 0),
    "S3": ("compact", "explicit", 0),
    "S1": ("compact", "implicit", 0),
    "S1a": ("ascii", "implicit", 0),
    "S2k4": ("compact", "ckpt", 4),
    "S2k8": ("compact", "ckpt", 8),
    "S2k16": ("compact", "ckpt", 16),
}
MODE_IDS = list(MODES)            # header byte = index; append-only

ACTIONS = sorted(set(L.VERBS.values()))
OPCODES = ["S", "U", "ENT", "QTY", "DEST", "TEMP", "DUR", "OUT", "CK"] + ACTIONS
_OPIDX = {o: i for i, o in enumerate(OPCODES)}
PREPS = sorted(L.PREPOSITIONS)
_PIDX = {p: i for i, p in enumerate(PREPS)}
VARIADIC = "*"

_SCHEMA = {
    "explicit": {"S": "", "U": "", "ENT": "IL", "QTY": "RNU", "DEST": "PR",
                 "TEMP": "NU", "DUR": "NU", "OUT": "R", "ACT": "R*"},
    "implicit": {"S": "", "U": "", "ENT": "L", "QTY": "INU", "DEST": "PI",
                 "TEMP": "NU", "DUR": "NU", "ACT": "I*"},
}
_SCHEMA["ckpt"] = dict(_SCHEMA["implicit"], CK="R*")


def _schema(state: str, op: str) -> str:
    sch = _SCHEMA[state]
    if op in ACTIONS:
        return sch["ACT"]
    if op not in sch:
        raise ValueError(f"opcode {op} not allowed in {state} streams")
    return sch[op]


# ------------------------------------------------------------ token helpers
def _int(tok: str) -> int:
    if not tok.isascii() or not tok.isdigit() or (len(tok) > 1 and tok[0] == "0"):
        raise ValueError(f"non-canonical integer {tok!r}")
    return int(tok)


def _ref(tok: str) -> tuple[int, int]:
    if not tok.startswith("e") or tok.count(".") != 1:
        raise ValueError(f"bad ref {tok!r}")
    a, b = tok[1:].split(".")
    return _int(a), _int(b)


def _eid(tok: str) -> int:
    if not tok.startswith("e"):
        raise ValueError(f"bad entity id {tok!r}")
    return _int(tok[1:])


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


# ------------------------------------------- explicit instrs <-> records
# A record is (opcode, tuple of typed values):
#   R -> (id, ver)   I -> id   L/N/U/P -> str

def to_records(instrs: list[Instr], state_mode: str, k: int = 0) -> list:
    """Explicit S0 instructions -> records for `state_mode`; validates by replay."""
    st = State()
    recs: list = []
    actions = 0
    for sent in _sentences(instrs):
        head = sent[0]
        recs.append((head.op, ()))
        if head.op == "U":
            if len(sent) != 1:
                raise ValueError("U sentence must be empty")
            continue
        body = sent[1:]
        ai = next((i for i, x in enumerate(body) if x.op in ACTIONS), None)
        if ai is None:
            raise ValueError("supported sentence without action")
        pre, act, post = body[:ai], body[ai], body[ai + 1:]
        nlit = 0
        while nlit < len(post) and post[nlit].op in ("DEST", "TEMP", "DUR"):
            nlit += 1
        lits, derived = post[:nlit], post[nlit:]

        def cur(eid: int, given: str | None = None) -> int:
            if eid >= len(st.entities):
                raise ValueError(f"reference to undeclared entity e{eid}")
            if given is not None and _ref(given) != (eid, st.entities[eid].ver):
                raise ValueError(f"stale or future version {given}")
            return eid

        for ins in pre:
            if ins.op == "ENT":
                eid = _eid(ins.args[0])
                if eid != len(st.entities):
                    raise ValueError("ENT ids must be sequential")
                st.new(ins.args[1])
                recs.append(("ENT", ((eid, ins.args[1]) if state_mode == "explicit"
                                     else (ins.args[1],))))
            elif ins.op == "QTY":
                eid = cur(_ref(ins.args[0])[0], ins.args[0])
                v = (st.entities[eid].ver,)
                recs.append(("QTY", (((eid, *v) if state_mode == "explicit" else eid),
                                     ins.args[1], ins.args[2])))
            else:
                raise ValueError(f"{ins.op} before action")
        arg_ids = [cur(_ref(a)[0], a) for a in act.args]
        dest_id = None
        lit_recs = []
        for ins in lits:
            if ins.op == "DEST":
                dest_id = cur(_ref(ins.args[1])[0], ins.args[1])
                val = ((dest_id, st.entities[dest_id].ver) if state_mode == "explicit"
                       else dest_id)
                lit_recs.append(("DEST", (ins.args[0], val)))
            else:
                lit_recs.append((ins.op, tuple(ins.args)))
        if state_mode == "explicit":
            recs.append((act.op, tuple((a, st.entities[a].ver) for a in arg_ids)))
        else:
            recs.append((act.op, tuple(arg_ids)))
        recs.extend(lit_recs)
        want = transition(st, act.op, arg_ids, dest_id)
        if [x.line() for x in want] != [x.line() for x in derived]:
            raise ValueError(f"derived lines disagree with state rules: "
                             f"{[x.line() for x in derived]} != "
                             f"{[x.line() for x in want]}")
        if state_mode == "explicit":
            for x in derived:
                if x.op == "ENT":
                    recs.append(("ENT", (_eid(x.args[0]), x.args[1])))
                else:
                    recs.append(("OUT", (_ref(x.args[0]),)))
        actions += 1
        if state_mode == "ckpt" and actions % k == 0:
            recs.append(("CK", tuple((i, e.ver) for i, e in enumerate(st.entities))))
    return recs


def from_records(recs: list, state_mode: str) -> list[Instr]:
    """Records -> canonical explicit instructions (replaying state rules)."""
    if state_mode == "explicit":
        instrs = []
        for op, vals in recs:
            sch = _schema("explicit", op)
            instrs.append(Instr(op, tuple(_fmt_explicit(t, v)
                                          for t, v in _typed(sch, vals))))
        to_records(instrs, "explicit")      # strict validation by replay
        return instrs

    st = State()
    instrs: list[Instr] = []
    pending = None          # (op, arg_ids, dest_id) awaiting derived lines
    actions = 0

    def close():
        nonlocal pending, actions
        if pending is not None:
            op, args, dest = pending
            instrs.extend(transition(st, op, args, dest))
            pending = None
            actions += 1

    def ref(eid: int) -> str:
        if not 0 <= eid < len(st.entities):
            raise ValueError(f"reference to undeclared entity e{eid}")
        return st.ref(eid)

    in_sentence = False
    saw_action = False
    for op, vals in recs:
        if op in ("S", "U"):
            if in_sentence and not saw_action:
                raise ValueError("supported sentence without action")
            close()
            instrs.append(Instr(op))
            in_sentence, saw_action = op == "S", False
            continue
        if op == "CK":
            close()
            if state_mode != "ckpt":
                raise ValueError("CK outside checkpoint mode")
            got = tuple(f"e{i}.{v}" for i, v in vals)
            if got != st.snapshot():
                raise ValueError(f"checkpoint mismatch: {got} != {st.snapshot()}")
            continue
        if not in_sentence:
            raise ValueError(f"{op} outside a supported sentence")
        if op == "ENT":
            if saw_action:
                raise ValueError("ENT after action")
            eid = st.new(vals[0])
            instrs.append(Instr("ENT", (f"e{eid}", vals[0])))
        elif op == "QTY":
            if saw_action:
                raise ValueError("QTY after action")
            instrs.append(Instr("QTY", (ref(vals[0]), vals[1], vals[2])))
        elif op in ACTIONS:
            if saw_action:
                raise ValueError("two actions in one sentence")
            saw_action = True
            args = list(vals)
            instrs.append(Instr(op, tuple(ref(a) for a in args)))
            pending = (op, args, None)
        elif op == "DEST":
            if not saw_action or pending is None or pending[2] is not None:
                raise ValueError("misplaced DEST")
            instrs.append(Instr("DEST", (vals[0], ref(vals[1]))))
            pending = (pending[0], pending[1], vals[1])
        elif op in ("TEMP", "DUR"):
            if not saw_action:
                raise ValueError(f"{op} before action")
            instrs.append(Instr(op, tuple(vals)))
        else:
            raise ValueError(f"unexpected {op}")
    if in_sentence and not saw_action:
        raise ValueError("supported sentence without action")
    close()
    return instrs


def _typed(sch: str, vals: tuple):
    if sch.endswith(VARIADIC):
        t = sch[0]
        return [(t, v) for v in vals]
    if len(sch) != len(vals):
        raise ValueError("arity mismatch")
    return list(zip(sch, vals))


def _fmt_explicit(t: str, v) -> str:
    if t == "R":
        return f"e{v[0]}.{v[1]}"
    if t == "I":
        return f"e{v}"
    return v


# --------------------------------------------------------------- encodings
def _ascii_write(recs, state_mode) -> bytes:
    lines = []
    for op, vals in recs:
        sch = _schema(state_mode, op)
        lines.append(" ".join([op] + [_fmt_explicit(t, v) for t, v in _typed(sch, vals)]))
    return "".join(l + "\n" for l in lines).encode("ascii")


def _ascii_read(b: bytes, state_mode) -> list:
    if not b:
        return []
    text = b.decode("ascii")
    if not text.endswith("\n"):
        raise ValueError("stream must end with newline")
    recs = []
    for line in text[:-1].split("\n"):
        parts = line.split(" ")
        op, toks = parts[0], parts[1:]
        if op not in _OPIDX:
            raise ValueError(f"unknown opcode {op!r}")
        sch = _schema(state_mode, op)
        if sch.endswith(VARIADIC):
            types = sch[0] * len(toks)
        else:
            types = sch
            if len(types) != len(toks):
                raise ValueError(f"arity mismatch in {line!r}")
        vals = []
        for t, tok in zip(types, toks):
            if t == "R":
                vals.append(_ref(tok))
            elif t == "I":
                vals.append(_eid(tok))
            else:
                if not tok or (t == "P" and tok not in _PIDX):
                    raise ValueError(f"bad operand {tok!r}")
                vals.append(tok)
        recs.append((op, tuple(vals)))
    return recs


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


def _compact_write(recs, state_mode) -> bytes:
    out = bytearray()
    for op, vals in recs:
        sch = _schema(state_mode, op)
        out.append(_OPIDX[op])
        if sch.endswith(VARIADIC):
            out += _varint(len(vals))
        for t, v in _typed(sch, vals):
            if t == "R":
                out += _varint(v[0]) + _varint(v[1])
            elif t == "I":
                out += _varint(v)
            elif t == "P":
                out.append(_PIDX[v])
            else:
                bb = v.encode("ascii")
                if not bb or b"\x00" in bb:
                    raise ValueError("bad string operand")
                out += bb + b"\x00"
    return bytes(out)


def _compact_read(b: bytes, state_mode) -> list:
    recs = []
    k = 0
    while k < len(b):
        oi = b[k]
        k += 1
        if oi >= len(OPCODES):
            raise ValueError("bad opcode")
        op = OPCODES[oi]
        sch = _schema(state_mode, op)
        if sch.endswith(VARIADIC):
            n, k = _rvarint(b, k)
            if n > len(b) - k:          # every operand needs >= 1 byte
                raise ValueError("operand count exceeds stream length")
            types = sch[0] * n
        else:
            types = sch
        vals = []
        for t in types:
            if t == "R":
                a, k = _rvarint(b, k)
                v, k = _rvarint(b, k)
                vals.append((a, v))
            elif t == "I":
                a, k = _rvarint(b, k)
                vals.append(a)
            elif t == "P":
                if k >= len(b) or b[k] >= len(PREPS):
                    raise ValueError("bad preposition")
                vals.append(PREPS[b[k]])
                k += 1
            else:
                z = b.find(b"\x00", k)
                if z <= k:
                    raise ValueError("bad string operand")
                vals.append(b[k:z].decode("ascii"))
                k = z + 1
        recs.append((op, tuple(vals)))
    return recs


# ----------------------------------------------------------------- public
def encode(instrs: list[Instr], mode: str) -> bytes:
    enc, state_mode, k = MODES[mode]
    recs = to_records(instrs, state_mode, k)
    return (_ascii_write if enc == "ascii" else _compact_write)(recs, state_mode)


def decode(s: bytes, mode: str) -> list[Instr]:
    enc, state_mode, k = MODES[mode]
    recs = (_ascii_read if enc == "ascii" else _compact_read)(s, state_mode)
    instrs = from_records(recs, state_mode)
    # canonical-form check: re-encoding must reproduce the stream exactly.
    # This pins checkpoint placement to every k-th action and rejects any
    # alternative spelling of the same content.
    if encode(instrs, mode) != s:
        raise ValueError("non-canonical semantic stream")
    return instrs


def symbols(s: bytes, mode: str) -> list[str]:
    """Model-facing symbol sequence for the semantic stream."""
    enc, state_mode, _ = MODES[mode]
    if not s:
        return []
    if enc == "ascii":
        toks = []
        for line in s.decode("ascii").splitlines():
            toks.extend(line.split(" "))
            toks.append("<NL>")
        return toks
    toks = []
    for op, vals in _compact_read(s, state_mode):
        toks.append(op)
        sch = _schema(state_mode, op)
        if sch.endswith(VARIADIC):
            toks.append(f"n{len(vals)}")
        for t, v in _typed(sch, vals):
            if t == "R":
                toks += [f"E{v[0]}", f"V{v[1]}"]
            elif t == "I":
                toks.append(f"E{v}")
            else:
                toks.append(f"{t}:{v}")
    return toks
