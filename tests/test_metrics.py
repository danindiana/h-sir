"""Coding models, rank fits, entropy estimators, BPE control, dedupe/split."""
import math
import random

import numpy as np

from hsir import control, dataset as ds, metrics, synthetic


def test_byte_model_is_a_proper_distribution():
    m = metrics.WBContextModel(2).fit([b"abcabcabd", b"hello world"])
    for ctx in (b"\x02\x02", b"ab", b"zz"):
        total = sum(2 ** -m.costs(ctx[:0] + bytes([b]))[0] for b in range(256))
        assert abs(total - 1) < 1e-9   # first symbol, BOS context
    c = m.costs(b"abcabc")
    assert np.all(c > 0) and np.all(np.isfinite(c))


def test_byte_model_learns():
    train = [t.encode() for t in synthetic.corpus(200, 5)]
    test = [t.encode() for t in synthetic.corpus(20, 6)]
    m0 = metrics.WBContextModel(0).fit(train)
    m3 = metrics.WBContextModel(3).fit(train)
    b0 = sum(m0.costs(x).sum() for x in test)
    b3 = sum(m3.costs(x).sum() for x in test)
    assert b3 < b0 < 8 * sum(len(x) for x in test)


def test_symbol_model_charges_unseen_symbols():
    m = metrics.WBSymbolModel(1).fit([["a", "b", "a", "b"]])
    seen = m.costs(["a"])[0]
    unseen = m.costs(["zzzz"])[0]
    assert unseen > seen + 8 * 4


def test_rank_frequency_prefers_true_family():
    rng = np.random.default_rng(0)
    r = np.arange(1, 2001)
    p = r ** -1.0
    p /= p.sum()
    draws = [f"t{i}" for i in rng.choice(r, 60000, p=p)]
    res = metrics.rank_frequency(draws[:40000], draws[40000:])
    assert res["status"] == "ok"
    assert abs(res["zipf"]["alpha"] - 1.0) < 0.1
    assert res["preferred_parametric_on_test"] == "zipf"


def test_conditional_entropy_detects_dependence():
    rnd = random.Random(0)
    pairs = [(c, f"{c}_{rnd.randint(0, 1)}") for c in
             (rnd.choice("abcdefgh") for _ in range(4000))]
    r = metrics.conditional_entropy(pairs[:3000], pairs[3000:])
    assert r["plugin_mm_bits"]["I(Y;C)"] > 2.5
    assert r["heldout_cross_entropy_bits"]["gain"] > 2.5


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(1)
    cols = {"a": rng.random(200) + 1, "b": rng.random(200) + 2}
    out = metrics.bootstrap(cols, {"ratio": lambda s: s["a"] / s["b"]}, 500, 0)
    lo, hi = out["ratio"]["ci95"]
    assert lo <= out["ratio"]["estimate"] <= hi


def test_bpe_is_reversible_and_hits_target():
    docs = [t.encode() for t in synthetic.corpus(300, 7)]
    bpe = control.BPE()
    fit = bpe.fit(docs, target_symbols_per_byte=0.5)
    assert fit["train_symbols_per_byte"] <= 0.5 + 1e-9
    for x in docs[:50] + [b"\xff\x00 odd \t bytes\n", b""]:
        assert bpe.decode(bpe.encode(x)) == x


def test_dedupe_and_split():
    base = synthetic.corpus(50, 8)
    docs = [ds.Doc(str(i), t.encode()) for i, t in enumerate(base)]
    docs.append(ds.Doc("dup-exact", base[0].upper().encode()))
    near = base[1] + " Serve."
    docs.append(ds.Doc("dup-near", near.encode()))
    kept, info = ds.dedupe(docs, seed=0, threshold=0.8)
    keys = {d.key for d in kept}
    assert "dup-exact" not in keys
    assert info["exact_duplicates_merged"] >= 1
    tr, te = ds.split(kept, 0.2, seed=0)
    assert len(tr) + len(te) == len(kept)
    assert not ({d.key for d in tr} & {d.key for d in te})
    tr2, te2 = ds.split(kept, 0.2, seed=0)
    assert [d.key for d in te] == [d.key for d in te2]
