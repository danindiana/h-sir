"""Determinism, canonical serialization, and versioned entity state."""
from hsir import codec, isa, parser, realizer, synthetic


def lines(text):
    return [i.line() for i in parser.parse(text).instrs]


def test_encoding_is_deterministic():
    for t in synthetic.corpus(100, seed=3):
        x = t.encode()
        assert codec.encode(x) == codec.encode(x)


def test_serialize_deserialize_inverse():
    for t in synthetic.corpus(100, seed=4):
        s = isa.serialize(parser.parse(t).instrs)
        assert isa.serialize(isa.deserialize(s)) == s


def test_versioned_state_transitions():
    got = lines("Mix the flour and the water. Heat the mixture. Stir it.")
    assert got == [
        "S", "ENT e0 flour", "ENT e1 water", "MIX e0.0 e1.0",
        "ENT e2 mixture", "OUT e2.0",
        "S", "HEAT e2.0", "OUT e2.1",
        "S", "STIR e2.1", "OUT e2.2",
    ]


def test_absorbing_creates_new_version_of_destination():
    got = lines("Mix eggs and milk. Add 2 cups flour to the mixture.")
    assert "ADD e3.0" in got and "DEST to e2.0" in got and "OUT e2.1" in got
    assert "QTY e3.0 2 cups" in got


def test_surface_variants_share_semantics():
    a = lines("Combine the flour and the sugar.")
    b = lines("Mix flour and sugar.")
    c = lines("blend flour, and sugar.")
    assert a == b == c


def test_verb_form_modifier_np():
    assert "ENT e1 brown_sugar" in lines("Whisk eggs and brown sugar.")
    assert "ENT e0 brown_sugar" in lines("Add the brown sugar.")


def test_realizer_is_function_of_S_only():
    s = isa.serialize(parser.parse("Preheat oven to 350 degrees. Bake for "
                                   "20 minutes.").instrs)
    assert realizer.realize(s) == (b"Preheat the oven to 350 degrees. "
                                   b"Bake for 20 minutes.")


def test_mode_depends_only_on_parseability():
    # long residual does not trigger a size-based fallback
    x = b"Chop the onions. " + b"Lorem ipsum dolor sit amet. " * 40
    assert codec.encode_parts(x).mode == codec.MODE_B
