"""Length-delimited binary container: E(x) = (S, C, R), with D(E(x)) == x.

Layout (format v3):
    MAGIC  b"HSIR"            4 bytes
    FORMAT_VERSION            1 byte   (3)
    MODE                      1 byte   (0 = A raw fallback, 1 = B predictive)
    SMODE                     1 byte   index into serial.MODE_IDS
    CSET                      1 byte   index into surface.CSET_IDS
    varint |x|, varint |S|, varint |C|, varint |R|
    S bytes, C bytes, R bytes
    CRC32(x)                  4 bytes, big-endian

Format v2 containers (no CSET byte, no C) still decode: they are C0.

Mode A: S and C are empty and R is x verbatim. Used when x is not strict UTF-8
or contains no supported sentence. No size-based switching.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

from . import parser, residual, serial, surface

MAGIC = b"HSIR"
FORMAT_VERSION = 3
DECODABLE_VERSIONS = (2, 3)
MODE_A, MODE_B = 0, 1


@dataclass
class Encoded:
    mode: int
    s: bytes
    r: bytes
    parse: parser.ParseResult | None
    smode: str = "S0"
    cset: str = "C0"
    c: bytes = b""
    cstats: dict = field(default_factory=dict)


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf: bytes, k: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        if k >= len(buf):
            raise ValueError("truncated varint")
        b = buf[k]
        k += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, k
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _check(smode: str, cset: str) -> None:
    if smode not in serial.MODES:
        raise ValueError(f"unknown serialization mode {smode!r}")
    if cset not in surface.CSETS:
        raise ValueError(f"unknown surface set {cset!r}")


def variant(x: bytes, pr: parser.ParseResult | None, smode: str = "S0",
            cset: str = "C0") -> Encoded:
    """Encode x from an existing parse (parse is mode-independent)."""
    _check(smode, cset)
    if pr is None or not pr.any_supported:
        return Encoded(MODE_A, b"", x, pr, smode, cset)
    s = serial.encode(pr.instrs, smode)
    c, cstats = surface.encode(pr.instrs, pr.surface, cset)
    y_hat = surface.realize(pr.instrs, cset, c).encode("ascii")
    r = residual.diff(y_hat, x)
    return Encoded(MODE_B, s, r, pr, smode, cset, c, cstats)


def encode_parts(x: bytes, smode: str = "S0", cset: str = "C0") -> Encoded:
    _check(smode, cset)
    try:
        text = x.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return Encoded(MODE_A, b"", x, None, smode, cset)
    return variant(x, parser.parse(text), smode, cset)


def with_smode(e: Encoded, smode: str) -> Encoded:
    """Re-serialize S in another mode; C and R do not depend on smode."""
    if e.mode == MODE_A:
        return Encoded(MODE_A, b"", e.r, e.parse, smode, e.cset)
    return Encoded(MODE_B, serial.encode(e.parse.instrs, smode), e.r, e.parse,
                   smode, e.cset, e.c, e.cstats)


def pack(x: bytes, e: Encoded) -> bytes:
    return (MAGIC + bytes([FORMAT_VERSION, e.mode,
                           serial.MODE_IDS.index(e.smode),
                           surface.CSET_IDS.index(e.cset)])
            + _varint(len(x)) + _varint(len(e.s)) + _varint(len(e.c))
            + _varint(len(e.r)) + e.s + e.c + e.r
            + zlib.crc32(x).to_bytes(4, "big"))


def pack_v2(x: bytes, e: Encoded) -> bytes:
    """Legacy writer, kept only to test backward decoding."""
    if e.c:
        raise ValueError("format v2 cannot carry surface choices")
    return (MAGIC + bytes([2, e.mode, serial.MODE_IDS.index(e.smode)])
            + _varint(len(x)) + _varint(len(e.s)) + _varint(len(e.r))
            + e.s + e.r + zlib.crc32(x).to_bytes(4, "big"))


def encode(x: bytes, smode: str = "S0", cset: str = "C0") -> bytes:
    return pack(x, encode_parts(x, smode, cset))


def unpack(buf: bytes) -> tuple[int, str, str, int, bytes, bytes, bytes, int]:
    if buf[:4] != MAGIC:
        raise ValueError("bad magic")
    if len(buf) < 7 or buf[4] not in DECODABLE_VERSIONS:
        raise ValueError("unsupported format version")
    ver, mode = buf[4], buf[5]
    if mode not in (MODE_A, MODE_B):
        raise ValueError("bad mode")
    if buf[6] >= len(serial.MODE_IDS):
        raise ValueError("bad serialization mode")
    smode = serial.MODE_IDS[buf[6]]
    if ver == 2:
        cset, k = "C0", 7
    else:
        if len(buf) < 8 or buf[7] >= len(surface.CSET_IDS):
            raise ValueError("bad surface set")
        cset, k = surface.CSET_IDS[buf[7]], 8
    n, k = _read_varint(buf, k)
    ns, k = _read_varint(buf, k)
    nc, k = (0, k) if ver == 2 else _read_varint(buf, k)
    nr, k = _read_varint(buf, k)
    if k + ns + nc + nr + 4 != len(buf):
        raise ValueError("length mismatch")
    s = buf[k:k + ns]
    c = buf[k + ns:k + ns + nc]
    r = buf[k + ns + nc:k + ns + nc + nr]
    crc = int.from_bytes(buf[k + ns + nc + nr:], "big")
    return mode, smode, cset, n, s, c, r, crc


def decode(buf: bytes) -> bytes:
    mode, smode, cset, n, s, c, r, crc = unpack(buf)
    if mode == MODE_A:
        if s or c:
            raise ValueError("mode A must have empty S and C")
        x = r
    else:
        instrs = serial.decode(s, smode)
        y_hat = surface.realize(instrs, cset, c).encode("ascii")
        x = residual.apply(y_hat, r)
    if len(x) != n or zlib.crc32(x) != crc:
        raise ValueError("integrity check failed")
    return x


def framing_bytes(x: bytes, e: Encoded) -> int:
    return len(pack(x, e)) - len(e.s) - len(e.c) - len(e.r)
