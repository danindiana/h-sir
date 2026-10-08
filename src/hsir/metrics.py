"""Coding-rate models, rank-frequency fits, conditional entropy, bootstrap.

Matched coding-model rule: every quantity that is compared (L(X) versus
L(S) + L(R|S)) is computed by the *same* model family, same order, trained on
the same train split, evaluated on the same test documents.
"""
from __future__ import annotations

import lzma
import math
from collections import Counter, defaultdict

import numpy as np
from scipy import optimize

BOS, SEP, EOR = 0x02, 0x1E, 0x1F
LOG2 = math.log(2)


# ---------------------------------------------------------------- byte model
class WBContextModel:
    """Static Witten-Bell interpolated order-k byte model.

    p_o(b|ctx) = (c(ctx,b) + T(ctx) p_{o-1}(b)) / (N(ctx) + T(ctx)),
    p_{-1}(b) = 1/256. Fit once on training streams; costs are held-out.
    """

    def __init__(self, order: int):
        self.k = order
        self.counts: list[dict[bytes, Counter]] = [defaultdict(Counter)
                                                   for _ in range(order + 1)]
        self._totals: list[dict[bytes, tuple[int, int]]] | None = None

    def fit(self, streams):
        k = self.k
        for s in streams:
            h = bytes([BOS]) * k + s
            for i in range(k, len(h)):
                b = h[i]
                for o in range(k + 1):
                    self.counts[o][h[i - o:i]][b] += 1
        self._totals = [{ctx: (sum(c.values()), len(c)) for ctx, c in lvl.items()}
                        for lvl in self.counts]
        return self

    def costs(self, s: bytes) -> np.ndarray:
        """Per-byte code length in bits."""
        k = self.k
        h = bytes([BOS]) * k + s
        out = np.empty(len(s))
        for i in range(k, len(h)):
            b = h[i]
            p = 1.0 / 256
            for o in range(k + 1):
                ctx = h[i - o:i]
                tot = self._totals[o].get(ctx)
                if tot is None:
                    break  # higher orders are unseen too
                n, t = tot
                p = (self.counts[o][ctx].get(b, 0) + t * p) / (n + t)
            out[i - k] = -math.log2(p)
        return out


class WBSymbolModel:
    """Witten-Bell order-k model over model-facing symbols (any hashables).

    Base distribution: uniform over train vocabulary + ESC; an unseen symbol
    pays ESC plus 8 bits per character of its spelling (+1 terminator), so
    novel lexemes are never free.
    """

    def __init__(self, order: int):
        self.k = order
        self.counts: list[dict[tuple, Counter]] = [defaultdict(Counter)
                                                   for _ in range(order + 1)]
        self.vocab: set = set()

    def fit(self, streams):
        k = self.k
        for s in streams:
            h = ("<BOS>",) * k + tuple(s)
            for i in range(k, len(h)):
                self.vocab.add(h[i])
                for o in range(k + 1):
                    self.counts[o][h[i - o:i]][h[i]] += 1
        self._tot = [{c: (sum(v.values()), len(v)) for c, v in lvl.items()}
                     for lvl in self.counts]
        self._base = 1.0 / (len(self.vocab) + 1)
        return self

    def costs(self, s) -> np.ndarray:
        k = self.k
        h = ("<BOS>",) * k + tuple(s)
        out = np.empty(len(h) - k)
        for i in range(k, len(h)):
            sym = h[i]
            known = sym in self.vocab
            p = self._base
            for o in range(k + 1):
                ctx = h[i - o:i]
                tot = self._tot[o].get(ctx)
                if tot is None:
                    break
                n, t = tot
                p = (self.counts[o][ctx].get(sym, 0) + t * p) / (n + t)
            bits = -math.log2(p)
            if not known:
                bits += 8 * (len(str(sym)) + 1)
            out[i - k] = bits
        return out


def baseline_symbols(x: bytes) -> list:
    return list(x) + ["<EOR>"]


def hybrid_symbols(mode: int, s_tokens: list, r: bytes,
                   c_tokens: list = ()) -> tuple[list, int, int]:
    """Model-facing serialization: S symbols, C symbols, then residual bytes.

    Returns (stream, cut_s, cut_c): stream[:cut_s] -> L(S),
    stream[cut_s:cut_c] -> L(C|S), stream[cut_c:] -> L(R|S,C).
    """
    toks = [f"<MODE{mode}>"] + list(s_tokens) + ["<SEP>"]
    cut_s = len(toks)
    toks += list(c_tokens) + ["<CSEP>"]
    cut_c = len(toks)
    return toks + list(r) + ["<EOR>"], cut_s, cut_c


def baseline_stream(x: bytes) -> bytes:
    return x + bytes([EOR])


CSEP = 0x1D


def hybrid_stream(mode: int, s: bytes, r: bytes, c: bytes = b"") -> tuple[bytes, int, int]:
    """Byte serialization; same three-way attribution as hybrid_symbols."""
    head = bytes([0x10 + mode]) + s + bytes([SEP])
    mid = c + bytes([CSEP])
    return head + mid + r + bytes([EOR]), len(head), len(head) + len(mid)


def lzma_bits(streams) -> int:
    data = b"".join(streams)
    return 8 * len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))


# ------------------------------------------------------------ rank-frequency
def _nll_zipf(alpha, ranks, counts, n):
    r = np.arange(1, n + 1, dtype=float)
    logz = np.log(np.sum(r ** -alpha))
    return float(np.sum(counts * (alpha * np.log(ranks) + logz)))


def _nll_sexp(params, ranks, counts, n):
    lam, beta = np.exp(params)
    r = np.arange(1, n + 1, dtype=float)
    e = -lam * r ** beta
    m = e.max()
    logz = m + np.log(np.sum(np.exp(e - m)))
    return float(np.sum(counts * (lam * ranks ** beta + logz)))


def rank_frequency(train: list[str], test: list[str]) -> dict:
    """Fit Zipf and stretched-exponential rank models on train, score on test.

    Ranks come from train frequencies; test tokens unseen in train map to a
    single OOV rank N+1 so every model assigns them mass.
    """
    tc = Counter(train)
    if len(tc) < 2:
        return {"status": "insufficient_types", "n_types": len(tc),
                "n_tokens": len(train)}
    order = [t for t, _ in sorted(tc.items(), key=lambda kv: (-kv[1], kv[0]))]
    rank = {t: i + 1 for i, t in enumerate(order)}
    n = len(order) + 1  # + OOV

    def agg(tokens):
        rc = Counter(rank.get(t, n) for t in tokens)
        rr = np.array(sorted(rc), dtype=float)
        cc = np.array([rc[int(x)] for x in rr], dtype=float)
        return rr, cc

    rtr, ctr = agg(train)
    rte, cte = agg(test)
    zt = optimize.minimize_scalar(_nll_zipf, bounds=(0.01, 6.0), method="bounded",
                                  args=(rtr, ctr, n))
    alpha = float(zt.x)
    st = optimize.minimize(_nll_sexp, x0=np.log([0.5, 0.5]), args=(rtr, ctr, n),
                           method="Nelder-Mead", options={"xatol": 1e-6,
                                                          "fatol": 1e-6})
    lam, beta = (float(v) for v in np.exp(st.x))
    ntest = max(1.0, cte.sum())
    # nonparametric reference: add-0.5 unigram over ranks
    probs = np.full(n, 0.5)
    for rr, cc in zip(rtr, ctr):
        probs[int(rr) - 1] += cc
    probs /= probs.sum()
    emp_nll = float(-np.sum(cte * np.log(probs[rte.astype(int) - 1])))
    res = {
        "status": "ok",
        "n_types_train": len(order),
        "n_tokens_train": int(ctr.sum()),
        "n_tokens_test": int(cte.sum()),
        "test_oov_rate": float(cte[rte == n].sum() / ntest) if len(cte) else 0.0,
        "zipf": {"alpha": alpha,
                 "train_bits_per_token": float(zt.fun / ctr.sum() / LOG2),
                 "test_bits_per_token": _nll_zipf(alpha, rte, cte, n) / ntest / LOG2},
        "stretched_exp": {"lambda": lam, "beta": beta,
                          "train_bits_per_token": float(st.fun / ctr.sum() / LOG2),
                          "test_bits_per_token": _nll_sexp(st.x, rte, cte, n) / ntest / LOG2},
        "empirical_add_half": {"test_bits_per_token": emp_nll / ntest / LOG2},
        "top_types": [[t, tc[t]] for t in order[:25]],
    }
    z, s = res["zipf"]["test_bits_per_token"], res["stretched_exp"]["test_bits_per_token"]
    res["preferred_parametric_on_test"] = "zipf" if z <= s else "stretched_exp"
    res["delta_bits_per_token_zipf_minus_sexp"] = z - s
    if len(order) < 50:
        res["caution"] = ("fewer than 50 types: asymptotic power-law inference "
                          "is not meaningful; read the full distribution")
    return res


# ------------------------------------------------------- conditional entropy
def _h_mm(counts: Counter) -> float:
    """Plug-in entropy (bits) with Miller-Madow correction."""
    n = sum(counts.values())
    if n == 0:
        return 0.0
    p = np.array(list(counts.values()), dtype=float) / n
    h = -np.sum(p * np.log2(p))
    return float(h + (len(counts) - 1) / (2 * n * LOG2))


def conditional_entropy(train_pairs: list[tuple[str, str]],
                        test_pairs: list[tuple[str, str]]) -> dict:
    """H(Y) vs H(Y|C) for (context, symbol) pairs, plug-in and held-out."""
    y = Counter(s for _, s in train_pairs)
    byc: dict[str, Counter] = defaultdict(Counter)
    for c, s in train_pairs:
        byc[c][s] += 1
    n = len(train_pairs)
    if n == 0:
        return {"status": "empty"}
    h_y = _h_mm(y)
    h_y_c = sum(sum(cnt.values()) / n * _h_mm(cnt) for cnt in byc.values())
    vocab = len(y) + 1
    uni = {s: (cnt + 0.5) / (n + 0.5 * vocab) for s, cnt in y.items()}
    oov = 0.5 / (n + 0.5 * vocab)

    def p_uni(s):
        return uni.get(s, oov)

    def p_cond(c, s):
        cnt = byc.get(c)
        if not cnt:
            return p_uni(s)
        nc, tc = sum(cnt.values()), len(cnt)
        return (cnt.get(s, 0) + tc * p_uni(s)) / (nc + tc)

    m = max(1, len(test_pairs))
    xh_y = -sum(math.log2(p_uni(s)) for _, s in test_pairs) / m
    xh_y_c = -sum(math.log2(p_cond(c, s)) for c, s in test_pairs) / m
    return {
        "status": "ok",
        "n_train": n, "n_test": len(test_pairs), "n_contexts": len(byc),
        "plugin_mm_bits": {"H(Y)": h_y, "H(Y|C)": h_y_c, "I(Y;C)": h_y - h_y_c},
        "heldout_cross_entropy_bits": {"H(Y)": xh_y, "H(Y|C)": xh_y_c,
                                       "gain": xh_y - xh_y_c},
        "note": ("plug-in values are biased low for large vocabularies; "
                 "held-out cross-entropies are upper bounds"),
    }


# ----------------------------------------------------------------- bootstrap
def bootstrap(per_doc: dict[str, np.ndarray], stats: dict, b: int, seed: int) -> dict:
    """Document-level percentile bootstrap of ratio statistics.

    `stats` maps name -> callable(sums: dict[str, float]) -> float, applied to
    column sums of a resample.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(per_doc.values())))
    if n == 0:
        return {}
    cols = {k: np.asarray(v, dtype=float) for k, v in per_doc.items()}
    draws = {name: np.empty(b) for name in stats}
    for i in range(b):
        idx = rng.integers(0, n, n)
        sums = {k: v[idx].sum() for k, v in cols.items()}
        for name, f in stats.items():
            draws[name][i] = f(sums)
    full = {k: v.sum() for k, v in cols.items()}
    return {name: {"estimate": float(f(full)),
                   "ci95": [float(np.percentile(draws[name], 2.5)),
                            float(np.percentile(draws[name], 97.5))],
                   "n_docs": n, "n_resamples": b}
            for name, f in stats.items()}
