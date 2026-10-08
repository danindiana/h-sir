"""Unsupported constructions fall back losslessly; nothing is fabricated."""
from hsir import codec, isa, parser


def test_non_utf8_is_mode_a_raw():
    x = b"Mix flour \xff and sugar."
    e = codec.encode_parts(x)
    assert e.mode == codec.MODE_A and e.s == b"" and e.r == x


def test_no_supported_sentence_is_mode_a():
    x = "If the batter is too thick, add milk. Enjoy!".encode()
    e = codec.encode_parts(x)
    assert e.mode == codec.MODE_A and e.r == x


def test_partial_support_marks_U_and_keeps_bytes():
    x = b"Chop the onions. Cook until golden, stirring occasionally."
    e = codec.encode_parts(x)
    assert e.mode == codec.MODE_B
    ops = [i.op for i in isa.deserialize(e.s)]
    assert ops.count("U") == 1
    assert b"until golden, stirring occasionally" in e.r
    assert codec.decode(codec.pack(x, e)) == x


def test_unsupported_sentence_leaves_entity_state_untouched():
    pr = parser.parse("Mix flour and sugar. Cook until golden. Stir it.")
    lines = [i.line() for i in pr.instrs]
    # "it" binds to the composite from MIX, not to anything in the U sentence
    assert "STIR e2.0" in lines and "OUT e2.1" in lines
    assert [s.supported for s in pr.sentences] == [True, False, True]


def test_parser_ignores_nothing_silently():
    # every sentence is either supported or U; counts must match
    text = "Preheat oven to 350 degrees. Mix A; B. Bake for 20 minutes. ???"
    pr = parser.parse(text)
    n_marks = sum(1 for i in pr.instrs if i.op in ("S", "U"))
    assert n_marks == len(pr.sentences)


def test_semantic_stream_is_ascii_even_for_unicode_source():
    e = codec.encode_parts("Mix the jalapeño and the flour. Add ٣ eggs.".encode())
    if e.s:
        e.s.decode("ascii")
