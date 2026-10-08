"""Entity-state machine shared by the encoder (parser) and every decoder.

There is exactly one implementation of the state-transition rules. The parser
calls `transition` to emit derived lines (composite ENT, OUT); implicit and
checkpointed decoders call the same function to regenerate them. Encoder and
decoder therefore cannot disagree about entity identity or version unless the
serialized stream itself is inconsistent, which `replay_explicit` detects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import lexicon as L
from .isa import Instr


@dataclass
class Entity:
    lexeme: str
    ver: int = 0
    composite: bool = False


@dataclass
class State:
    entities: list[Entity] = field(default_factory=list)
    by_lexeme: dict[str, int] = field(default_factory=dict)
    focus: int | None = None       # most recent composite
    last: int | None = None        # most recently touched entity

    def ref(self, eid: int) -> str:
        return f"e{eid}.{self.entities[eid].ver}"

    def new(self, lexeme: str, composite: bool = False) -> int:
        eid = len(self.entities)
        self.entities.append(Entity(lexeme, 0, composite))
        if not composite:
            self.by_lexeme[lexeme] = eid
        return eid

    def copy(self) -> "State":
        return State([Entity(e.lexeme, e.ver, e.composite) for e in self.entities],
                     dict(self.by_lexeme), self.focus, self.last)

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self.ref(i) for i in range(len(self.entities)))


def transition(st: State, op: str, arg_ids: list[int],
               dest_id: int | None) -> list[Instr]:
    """Apply the state rules for one action; return the derived lines.

    Derived lines are always: optional composite `ENT`, then `OUT` lines.
    """
    out: list[Instr] = []
    if op in L.COMBINING and len(arg_ids) >= 2:
        eid = st.new(L.COMPOSITE_LEXEME, composite=True)
        out.append(Instr("ENT", (f"e{eid}", L.COMPOSITE_LEXEME)))
        out.append(Instr("OUT", (st.ref(eid),)))
        st.focus = st.last = eid
    elif op in L.ABSORBING and dest_id is not None:
        st.entities[dest_id].ver += 1
        out.append(Instr("OUT", (st.ref(dest_id),)))
        st.last = dest_id
        if st.entities[dest_id].composite or len(arg_ids) > 0:
            st.entities[dest_id].composite = True
            st.focus = dest_id
    elif op in L.TRANSFORMING or (op in L.COMBINING and len(arg_ids) == 1):
        for eid in dict.fromkeys(arg_ids):
            st.entities[eid].ver += 1
            out.append(Instr("OUT", (st.ref(eid),)))
            st.last = eid
    elif arg_ids:
        st.last = arg_ids[-1]
    return out
