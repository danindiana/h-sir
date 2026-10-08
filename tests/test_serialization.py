"""Serialization modes S0/S1/S1a/S2k*/S3: exactness, canonicality, tamper detection."""
import random

import pytest
from hypothesis import given, settings, strategies as st

from hsir import codec, isa, parser, serial, synthetic

MODES = list(serial.MODES)
DOCS = [t for t in synthetic.corpus(300, seed=11)]


def lines(instrs):
    return [i.line() for i in instrs]


@pytest.mark.parametrize("mode", MODES)
def test_instruction_roundtrip_all_modes(mode):
    for t in DOCS:
        pr = parser.parse(t)
        s = serial.encode(pr.instrs, mode)
        assert lines(serial.decode(s, mode)) == lines(pr.instrs)


@pytest.mark.parametrize("mode", MODES)
def test_container_roundtrip_all_modes(mode):
    for t in DOCS[:100] + ["", "Mix \xff".encode("latin-1").decode("latin-1")]:
        x = t.encode("utf-8", "surrogateescape")
        assert codec.decode(codec.encode(x, mode)) == x


@settings(max_examples=300, deadline=None)
@given(st.binary(max_size=200), st.sampled_from(MODES))
def test_arbitrary_bytes_any_mode(x, mode):
    assert codec.decode(codec.encode(x, mode)) == x


def test_s0_is_byte_identical_to_v031_format():
    for t in DOCS:
        pr = parser.parse(t)
        assert serial.encode(pr.instrs, "S0") == isa.serialize(pr.instrs)


def test_residual_identical_across_modes():
    for t in DOCS[:100]:
        x = t.encode()
        rs = {codec.encode_parts(x, m).r for m in MODES}
        assert len(rs) == 1


def test_implicit_modes_carry_no_derived_lines():
    pr = parser.parse("Mix the flour and the water. Heat the mixture. Stir it.")
    s1a = serial.encode(pr.instrs, "S1a").decode()
    assert "OUT" not in s1a and "mixture" not in s1a and ".0" not in s1a
    assert s1a == "S\nENT flour\nENT water\nMIX e0 e1\nS\nHEAT e2\nS\nSTIR e2\n"


def test_checkpoint_schedule():
    text = " ".join(["Stir the flour."] * 9)
    pr = parser.parse(text)
    for k in (4, 8):
        recs = serial.to_records(pr.instrs, "ckpt", k)
        cks = [i for i, (op, _) in enumerate(recs) if op == "CK"]
        assert len(cks) == 9 // k
        assert recs[cks[0]][1] == ((0, k),)       # flour version == k


def _tamper_ckpt(s: bytes) -> bytes:
    recs = serial._compact_read(s, "ckpt")
    out = []
    for op, vals in recs:
        if op == "CK":
            vals = tuple((i, v + 1) for i, v in vals)
        out.append((op, vals))
    return serial._compact_write(out, "ckpt")


def test_checkpoint_mismatch_detected():
    pr = parser.parse(" ".join(["Stir the flour."] * 8))
    s = serial.encode(pr.instrs, "S2k4")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        serial.decode(_tamper_ckpt(s), "S2k4")


def test_missing_checkpoint_is_noncanonical():
    pr = parser.parse(" ".join(["Stir the flour."] * 8))
    recs = [r for r in serial.to_records(pr.instrs, "ckpt", 4) if r[0] != "CK"]
    s = serial._compact_write(recs, "ckpt")
    with pytest.raises(ValueError, match="non-canonical"):
        serial.decode(s, "S2k4")


def test_explicit_stream_with_wrong_version_rejected():
    s = b"S\nENT e0 flour\nHEAT e0.0\nOUT e0.2\n"     # should be e0.1
    with pytest.raises(ValueError):
        serial.decode(s, "S0")
    s = b"S\nENT e0 flour\nHEAT e0.1\nOUT e0.2\n"     # stale/future input ref
    with pytest.raises(ValueError):
        serial.decode(s, "S0")


def test_undeclared_entity_rejected():
    with pytest.raises(ValueError):
        serial.decode(b"S\nHEAT e3\n", "S1a")
    with pytest.raises(ValueError):
        serial.decode(b"S\nENT e1 flour\nHEAT e1.0\nOUT e1.1\n", "S0")


def test_explicit_missing_composite_rejected():
    s = b"S\nENT e0 flour\nENT e1 water\nMIX e0.0 e1.0\n"   # no derived ENT/OUT
    with pytest.raises(ValueError, match="derived lines"):
        serial.decode(s, "S0")


@settings(max_examples=300, deadline=None)
@given(st.binary(max_size=60), st.sampled_from(MODES))
def test_garbage_streams_never_decode_silently_wrong(b, mode):
    # any accepted stream must be canonical: re-encoding reproduces it
    try:
        instrs = serial.decode(b, mode)
    except (ValueError, UnicodeDecodeError, KeyError):
        return
    assert serial.encode(instrs, mode) == b


def test_mode_header_selects_decoder():
    x = b"Mix the flour and the water. Bake for 20 minutes."
    for m in MODES:
        buf = codec.encode(x, m)
        assert codec.unpack(buf)[1] == m
    buf = bytearray(codec.encode(x, "S1"))
    buf[6] = serial.MODE_IDS.index("S0")     # lie about the mode
    with pytest.raises(Exception):
        codec.decode(bytes(buf))


def test_mixture_anaphor_without_composite_is_new_entity():
    got = lines(parser.parse("Preheat oven to 350 degrees. Stir the mixture.").instrs)
    assert "ENT e1 mixture" in got and "STIR e1.0" in got
    got = lines(parser.parse("Chop the onions. Stir it.").instrs)
    assert "STIR e0.1" in got                # pronoun keeps last-touched fallback
