"""Surface-factorized residuals (C sets): exactness, canonicality, backward compat."""
import random

import pytest
from hypothesis import given, settings, strategies as st

from hsir import codec, parser, realizer, serial, surface, synthetic

CSETS = list(surface.CSETS)
SMODES = list(serial.MODES)
DOCS = synthetic.corpus(200, seed=21)


def test_c0_render_identical_to_v032_realizer():
    for t in synthetic.corpus(500, seed=22):
        pr = parser.parse(t)
        assert surface.render(pr.instrs, ())[0] == realizer.realize_instrs(pr.instrs)
        c, _ = surface.encode(pr.instrs, pr.surface, "C0")
        assert c == b""


@pytest.mark.parametrize("cset", CSETS)
def test_every_cset_roundtrips(cset):
    for t in DOCS:
        x = t.encode()
        assert codec.decode(codec.encode(x, "S1", cset)) == x


def test_every_arm_combination_roundtrips():
    rng = random.Random(0)
    for t in DOCS[:60]:
        x = t.encode()
        for sm in SMODES:
            cs = rng.choice(CSETS)
            assert codec.decode(codec.encode(x, sm, cs)) == x


@settings(max_examples=300, deadline=None)
@given(st.binary(max_size=200), st.sampled_from(SMODES), st.sampled_from(CSETS))
def test_arbitrary_bytes_any_arm(x, sm, cs):
    assert codec.decode(codec.encode(x, sm, cs)) == x


def test_v2_containers_still_decode():
    for t in DOCS[:50] + ["", "\xff"]:
        x = t.encode("utf-8", "surrogateescape")
        for sm in SMODES:
            e = codec.encode_parts(x, sm, "C0")
            assert codec.decode(codec.pack_v2(x, e)) == x


def test_choices_shrink_residual():
    x = b"Combine flour, sugar and the butter in a bowl.\nadd it to a pan."
    r0 = codec.encode_parts(x, "S1", "C0").r
    e4 = codec.encode_parts(x, "S1", "C4")
    assert len(e4.r) < len(r0)
    assert surface.realize(e4.parse.instrs, "C4", e4.c).encode() == x


def test_observed_choices():
    pr = parser.parse("Combine flour, sugar and the butter in a bowl.\n"
                      "add it to a pan.")
    o = pr.surface
    assert o[0]["verb"] == "combine" and o[0]["list"] == "plain"
    assert [n["det"] for n in o[0]["nps"]] == ["", "", "the"]
    assert o[0]["dest"]["det"] == "a"
    assert o[1]["case"] == "lower" and o[1]["sep_before"] == "\n"
    assert o[1]["nps"][0]["ref"] == "it"


def test_only_non_default_choices_are_stored():
    pr = parser.parse("Mix the flour and the sugar. Bake the mixture.")
    for cs in CSETS:
        c, stats = surface.encode(pr.instrs, pr.surface, cs)
        assert c == b""                  # source already matches defaults
        assert all(v["emitted"] == 0 for v in stats.values())


def test_permuted_ids_realize_identically():
    for t in DOCS[:80]:
        pr = parser.parse(t)
        if not pr.any_supported:
            continue
        c4, _ = surface.encode(pr.instrs, pr.surface, "C4")
        c4p, _ = surface.encode(pr.instrs, pr.surface, "C4P")
        assert len(c4) == len(c4p)
        assert (surface.realize(pr.instrs, "C4", c4)
                == surface.realize(pr.instrs, "C4P", c4p))


def test_bad_surface_streams_rejected():
    pr = parser.parse("Mix the flour and the sugar.")
    with pytest.raises(ValueError, match="default"):
        surface.decode(pr.instrs, b"\x00\x00", "C4")          # stores default
    with pytest.raises(ValueError, match="out of range"):
        surface.decode(pr.instrs, b"\x00\x7f", "C4")
    with pytest.raises(ValueError, match="beyond last slot"):
        surface.decode(pr.instrs, b"\x7f\x01", "C4")
    with pytest.raises(ValueError):
        surface.realize(pr.instrs, "C0", b"\x00\x01")         # C0 carries nothing


@settings(max_examples=300, deadline=None)
@given(st.binary(max_size=24), st.sampled_from(CSETS[1:]))
def test_garbage_surface_streams_fail_cleanly(c, cs):
    pr = parser.parse("Combine flour, sugar and the butter in a bowl. "
                      "Add it to a pan. Bake for 20 minutes.")
    try:
        surface.realize(pr.instrs, cs, c)
    except ValueError:
        pass


def test_cset_header_roundtrip_and_lie_detected():
    x = b"combine flour and sugar.\nadd it to a pan."
    for cs in CSETS:
        buf = codec.encode(x, "S1", cs)
        assert codec.unpack(buf)[2] == cs
    buf = bytearray(codec.encode(x, "S1", "C4"))
    buf[7] = surface.CSET_IDS.index("C1")
    with pytest.raises(Exception):
        codec.decode(bytes(buf))
