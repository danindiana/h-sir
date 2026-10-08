"""Semantic instruction stream (S): line-oriented canonical ASCII format.

One instruction per line, fields separated by single spaces, lines ended by
'\n'.  See spec/semantic_isa.md for the grammar.

Line kinds (opcode = first field):
    S                       sentence boundary (supported sentence follows)
    U                       unsupported sentence (realizer emits nothing)
    ENT <eid> <lexeme>      introduce entity identity
    QTY <ref> <num> <unit>  quantity attached to an argument in this sentence
    <OP> <ref>*             action operator over versioned entity refs
    DEST <prep> <ref>       destination / location of the action
    TEMP <num> <unit>       temperature literal
    DUR <num> <unit>        duration literal
    OUT <ref>               resulting entity state (new version)

A ref is e<id>.<version>; an eid is e<id>.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STRUCTURAL = {"S", "U", "ENT", "QTY", "DEST", "TEMP", "DUR", "OUT"}


@dataclass(frozen=True)
class Instr:
    op: str
    args: tuple[str, ...] = ()

    def line(self) -> str:
        return " ".join((self.op,) + self.args)


def serialize(instrs: list[Instr]) -> bytes:
    out = "".join(i.line() + "\n" for i in instrs)
    b = out.encode("ascii")  # raises if a non-ASCII operand slipped in
    return b


def deserialize(s: bytes) -> list[Instr]:
    if not s:
        return []
    text = s.decode("ascii")
    if not text.endswith("\n"):
        raise ValueError("semantic stream must end with newline")
    instrs = []
    for line in text[:-1].split("\n"):
        parts = line.split(" ")
        if not parts or not parts[0]:
            raise ValueError(f"empty instruction line: {line!r}")
        instrs.append(Instr(parts[0], tuple(parts[1:])))
    return instrs


def is_ref(tok: str) -> bool:
    if not tok.startswith("e") or "." not in tok:
        return False
    a, _, b = tok[1:].partition(".")
    return a.isdigit() and b.isdigit()


def classify_operand(op: str, idx: int, tok: str) -> str:
    """Operand category for the Zipf/H(operand|operator) analysis."""
    if is_ref(tok) or (op == "ENT" and idx == 0):
        return "entity_ref"
    if op == "ENT" and idx == 1:
        return "concept"
    if op in ("QTY", "TEMP", "DUR"):
        # QTY <ref> <num> <unit>, TEMP/DUR <num> <unit>
        if tok[:1].isdigit():
            return "literal"
        if tok == "-":
            return "literal"
        return "unit"
    if op == "DEST" and idx == 0:
        return "relation"
    return "other"
