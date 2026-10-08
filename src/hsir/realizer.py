"""Deterministic surface realizer: semantic stream S -> canonical text y_hat.

y_hat is a function of S alone (plus the frozen rule tables). The residual
is an edit script from y_hat to the source bytes, so it is conditioned on S
by construction.
"""
from __future__ import annotations

from .isa import Instr


def realize_instrs(instrs: list[Instr]) -> str:
    """Default (C0) realization; one code path shared with surface.render."""
    from . import surface
    return surface.render(instrs, ())[0]


def realize(s: bytes, smode: str = "S0") -> bytes:
    from . import serial
    return realize_instrs(serial.decode(s, smode)).encode("ascii")
