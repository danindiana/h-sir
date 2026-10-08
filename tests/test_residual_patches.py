"""Residual edit scripts: exactness, canonical numbers, bounds checks."""
import pytest
from hypothesis import given, settings, strategies as st

from hsir import residual


@settings(max_examples=500, deadline=None)
@given(st.binary(max_size=200), st.binary(max_size=200))
def test_diff_apply_inverse(y, x):
    assert residual.apply(y, residual.diff(y, x)) == x


def test_identity_is_empty():
    assert residual.diff(b"abc", b"abc") == b""


def test_payload_may_contain_framing_bytes():
    y = b"Mix the flour."
    x = b"Mix R1,2,3: the D4; flour I0,0:."
    assert residual.apply(y, residual.diff(y, x)) == x


def test_short_equal_runs_are_coalesced():
    # an inserted sentence must not be shredded into many tiny ops
    y = b"Chop the onions. Stir the mixture."
    x = b"Chop the onions.\nLet stand 5 minutes before cutting.\nStir it."
    r = residual.diff(y, x)
    assert residual.apply(y, r) == x
    ops = sum(r.count(t) for t in (b"R", b"I", b"D"))  # upper bound incl payload
    assert ops <= 6


@pytest.mark.parametrize("bad", [b"X1,2;", b"D1;", b"R1,1:", b"I01,1:a",
                                 b"D99,1;", b"I0,5:ab", b"R0,1,1"])
def test_malformed_residual_rejected(bad):
    with pytest.raises(ValueError):
        residual.apply(b"abc", bad)
