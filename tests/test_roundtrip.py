"""D(E(x)) == x for every input: synthetic recipes, arbitrary bytes, edge cases."""
import pytest
from hypothesis import given, settings, strategies as st

from hsir import codec, synthetic


@pytest.mark.parametrize("text", synthetic.corpus(300, seed=1))
def test_synthetic_roundtrip(text):
    x = text.encode()
    assert codec.decode(codec.encode(x)) == x


EDGE = [
    b"", b".", b"\n\n\n", b"   ", b"\x00", b"\xff\xfe\xfd",
    "Préchauffez le four à 180°C.".encode(),
    b"Mix flour.\r\nBake for 10 minutes.\r\n",
    b"Mix flour \xff and sugar.",            # invalid UTF-8 -> mode A
    "Mix the 🍓 and cream.".encode(),
    b"Add 2 eggs to the mixture." * 50,
    b"Stir it. Stir it. Stir it.",
    b"HSIR\x01\x01 fake container bytes",
    "Add ٣ eggs.".encode(),                 # non-ASCII digit
    b"Mix flour,, sugar.",
    b"Bake at 350\xc2\xb0F for 1-2 hours.",
]


@pytest.mark.parametrize("x", EDGE)
def test_edge_roundtrip(x):
    assert codec.decode(codec.encode(x)) == x


@settings(max_examples=400, deadline=None)
@given(st.binary(max_size=400))
def test_arbitrary_bytes_roundtrip(x):
    assert codec.decode(codec.encode(x)) == x


_words = st.sampled_from(
    ["mix", "the", "flour", "and", "sugar", "bake", "for", "10", "minutes",
     "add", "to", "it", "mixture", "at", "350", "degrees", ",", ".", "\n",
     "Stir", "Preheat", "oven", "2", "cups", "in", "a", "bowl", "brown"])


@settings(max_examples=400, deadline=None)
@given(st.lists(_words, max_size=60), st.sampled_from([" ", "", "  "]))
def test_recipe_like_text_roundtrip(ws, sep):
    x = sep.join(ws).encode()
    assert codec.decode(codec.encode(x)) == x


def test_corruption_detected():
    x = b"Mix the flour and the sugar. Bake for 20 minutes."
    buf = bytearray(codec.encode(x))
    buf[-1] ^= 0x01                          # flip CRC bit
    with pytest.raises(ValueError):
        codec.decode(bytes(buf))
    with pytest.raises(ValueError):
        codec.decode(codec.encode(x)[:-5])   # truncation
