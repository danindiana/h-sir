"""Byte-exact reversible edit script: y_hat -> x.

Wire format (ASCII framing, raw payload bytes):
    op := 'R' skip ',' del ',' ins ':' <ins raw bytes>
        | 'D' skip ',' del ';'
        | 'I' skip ',' ins ':' <ins raw bytes>
skip/del/ins are decimal. `skip` counts bytes of y_hat copied unchanged since
the previous op; trailing y_hat bytes after the last op are copied
implicitly. Payload bytes are arbitrary (length-delimited, never parsed).
"""
from __future__ import annotations

from difflib import SequenceMatcher


# Equal runs shorter than this, sandwiched between edits, are absorbed into a
# single replace. A separate op costs ~5-8 framing bytes, so tiny spurious
# matches (e.g. stray letters aligned inside an unsupported sentence) cost more
# than they save. Data-independent codec parameter; changing it changes R only.
MIN_EQUAL = 6


def _coalesce(ops):
    out = []
    for op in ops:
        tag, i1, i2, j1, j2 = op
        if (out and tag == "equal" and i2 - i1 < MIN_EQUAL):
            out.append(op)  # tentatively keep; resolved below
            continue
        if tag != "equal" and len(out) >= 2 and out[-1][0] == "equal" \
                and out[-1][2] - out[-1][1] < MIN_EQUAL and out[-2][0] != "equal":
            eq = out.pop()
            prev = out.pop()
            out.append(("replace", prev[1], i2, prev[3], j2))
            continue
        if tag != "equal" and out and out[-1][0] != "equal":
            prev = out.pop()
            out.append(("replace", prev[1], i2, prev[3], j2))
            continue
        out.append(op)
    return out


def diff(y_hat: bytes, x: bytes) -> bytes:
    sm = SequenceMatcher(None, y_hat, x, autojunk=False)
    out = bytearray()
    pos = 0
    for tag, i1, i2, j1, j2 in _coalesce(sm.get_opcodes()):
        if tag == "equal":
            continue
        skip = i1 - pos
        if tag == "replace" and i2 == i1:
            tag = "insert"
        elif tag == "replace" and j2 == j1:
            tag = "delete"
        if tag == "replace":
            out += f"R{skip},{i2 - i1},{j2 - j1}:".encode() + x[j1:j2]
        elif tag == "delete":
            out += f"D{skip},{i2 - i1};".encode()
        elif tag == "insert":
            out += f"I{skip},{j2 - j1}:".encode() + x[j1:j2]
        pos = i2
    return bytes(out)


def _num(r: bytes, k: int, term: bytes) -> tuple[int, int]:
    j = k
    while j < len(r) and 48 <= r[j] <= 57:
        j += 1
    if j == k or j >= len(r) or r[j:j + 1] != term:
        raise ValueError(f"malformed residual at {k}")
    if j - k > 1 and r[k] == 48:
        raise ValueError("non-canonical number")
    return int(r[k:j]), j + 1


def apply(y_hat: bytes, r: bytes) -> bytes:
    out = bytearray()
    pos = 0
    k = 0
    while k < len(r):
        tag = r[k:k + 1]
        k += 1
        if tag == b"R":
            skip, k = _num(r, k, b",")
            dl, k = _num(r, k, b",")
            il, k = _num(r, k, b":")
        elif tag == b"D":
            skip, k = _num(r, k, b",")
            dl, k = _num(r, k, b";")
            il = 0
        elif tag == b"I":
            skip, k = _num(r, k, b",")
            il, k = _num(r, k, b":")
            dl = 0
        else:
            raise ValueError(f"bad residual tag at {k - 1}")
        if pos + skip + dl > len(y_hat) or k + il > len(r):
            raise ValueError("residual offset out of range")
        out += y_hat[pos:pos + skip]
        pos += skip + dl
        out += r[k:k + il]
        k += il
    out += y_hat[pos:]
    return bytes(out)
