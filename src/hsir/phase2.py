"""Phase II feasibility report: representation economics before any training.

Usage:
    python -m hsir.phase2 --input recipes.csv --out phase2/ [--limit N]
    python -m hsir.phase2 --synthetic 2000 --out phase2_synth/

Every preregistered arm (S serialization mode x surface set) is scored on the
same parse, documents, split, coders and bootstrap indices. The hybrid code is
charged in full: L(S) + L(C|S) + L(R|S,C).

Writes the eleven-file contract in spec/codec_contract.md. A synthetic run
always yields decision PENDING_REAL_DATA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from . import codec, figures, isa, lexicon, metrics, serial, surface, synthetic
from . import dataset as ds
from .control import BPE
from .parser import _SENT_RE

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent.parent
PREREG_PATH = ROOT / "spec" / "prereg.json"
COMPILER_SOURCES = ["lexicon.py", "isa.py", "state.py", "parser.py",
                    "realizer.py", "serial.py", "surface.py", "residual.py",
                    "codec.py"]
SEVERITY = {"GO": 0, "REVISE": 1, "STOP": 2}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _dump(out: Path, name: str, obj) -> None:
    with open(out / name, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=float)
        f.write("\n")


def _analyse(instrs: list[isa.Instr]):
    """Operator tokens, (category, operand) tokens and (action, concept) pairs
    from the canonical (explicit) instruction list."""
    ops, operands, ctx_pairs = [], [], []
    sent: list[isa.Instr] = []

    def flush():
        action = next((i.op for i in sent if i.op not in isa.STRUCTURAL), None)
        for ins in sent:
            if ins.op == "ENT" and action:
                ctx_pairs.append((action, ins.args[1]))
        sent.clear()

    for ins in instrs:
        if ins.op in ("S", "U"):
            flush()
        ops.append(ins.op)
        for j, a in enumerate(ins.args):
            operands.append((isa.classify_operand(ins.op, j, a), a))
        sent.append(ins)
    flush()
    return ops, operands, ctx_pairs


def _decide(fam_inputs: dict, bcov: float, thr: dict) -> dict:
    """Preregistered GO / REVISE / STOP for one (mode, family)."""
    stop, revise = [], []
    g_ci, g_whole = fam_inputs["gamma_whole_ci95"], fam_inputs["gamma_whole_no_shared"]
    g_cov, g_target = fam_inputs["gamma_covered_subset"], fam_inputs["gamma_at_prereg_target_with_shared"]
    rho = fam_inputs["rho_R"]
    if g_ci[0] > thr["gamma_stop_ci_low"]:
        stop.append(f"gamma CI low {g_ci[0]:.3f} > {thr['gamma_stop_ci_low']}")
    if bcov < thr["byte_coverage_stop_min"]:
        stop.append(f"byte coverage {bcov:.3f} < {thr['byte_coverage_stop_min']}")
    if g_cov is not None and g_cov < 1.0 <= g_whole:
        stop.append(f"gain only on covered subset (gamma_covered {g_cov:.3f}, "
                    f"gamma_whole {g_whole:.3f})")
    if rho > thr["rho_R_diagnostic_max"]:
        revise.append(f"rho_R {rho:.3f} > {thr['rho_R_diagnostic_max']}")
    if g_target > thr["gamma_revise_max"]:
        revise.append(f"gamma at target scale {g_target:.3f} > {thr['gamma_revise_max']}")
    if bcov < thr["byte_coverage_revise_min"]:
        revise.append(f"byte coverage {bcov:.3f} < {thr['byte_coverage_revise_min']}")
    return {"decision": "STOP" if stop else ("REVISE" if revise else "GO"),
            "reasons": stop + revise}


def run(docs: list[ds.Doc], load_audit: dict, out: Path, prereg: dict,
        source_desc: dict, synthetic_run: bool,
        prereg_path: Path = PREREG_PATH) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    seed = prereg["seed"]
    thr = prereg["thresholds"]
    arms: dict[str, tuple[str, str]] = {
        name: tuple(v) for name, v in prereg.get("arms", {"S0": ["S0", "C0"]}).items()}
    for name, (sm, cs) in arms.items():
        if sm not in serial.MODES or cs not in surface.CSETS:
            raise ValueError(f"arm {name}: unknown mode {sm!r} or surface set {cs!r}")
    names = list(arms)
    csets = sorted({cs for _, cs in arms.values()}, key=surface.CSET_IDS.index)
    kord = prereg["primary_order"]
    pk = f"order_{kord}"

    # ------------------------------------------------ audit, dedupe, split
    kept, dd = ds.dedupe(docs, seed, prereg["near_dup_jaccard"])
    train, test = ds.split(kept, prereg["test_fraction"], seed)
    audit = {"load": load_audit, "dedupe": dd,
             "split": {"train_docs": len(train), "test_docs": len(test),
                       "train_bytes": sum(len(d.x) for d in train),
                       "test_bytes": sum(len(d.x) for d in test),
                       "unit": "document", "test_fraction": prereg["test_fraction"]},
             "fields_used": ds.FIELDS_USED,
             "privileged_fields_excluded": ["title", "ingredients", "NER",
                                            "link", "source"],
             "synthetic": synthetic_run}
    _dump(out, "dataset_audit.json", audit)

    # ------------------------------------------------ encode + round trip
    failures, unsupported_examples = [], []
    enc: dict[int, dict[str, codec.Encoded]] = {}
    rt = {a: Counter() for a in names}
    reasons = Counter()
    mode_counts = Counter()
    for split_name, part in (("train", train), ("test", test)):
        for d in part:
            base = codec.encode_parts(d.x, "S0", "C0")
            by_cset = {cs: base if cs == "C0" else
                       codec.variant(d.x, base.parse if base.mode == codec.MODE_B
                                     else None, "S0", cs)
                       for cs in csets}
            enc[id(d)] = {}
            for a in names:
                sm, cs = arms[a]
                e = codec.with_smode(by_cset[cs], sm)
                enc[id(d)][a] = e
                try:
                    ok = codec.decode(codec.pack(d.x, e)) == d.x
                    err = "" if ok else "mismatch"
                except Exception as ex:  # reported, not raised
                    ok, err = False, f"{type(ex).__name__}: {ex}"
                rt[a]["ok" if ok else "fail"] += 1
                if not ok:
                    failures.append({"kind": "roundtrip", "arm": a,
                                     "key": d.key, "error": err})
            mode_counts["mode_B" if base.mode == codec.MODE_B else "mode_A"] += 1
            if base.parse:
                sents = base.parse.sentences
                for st in sents:
                    if not st.supported:
                        reasons[st.reason.split(" '")[0]] += 1
                if len(unsupported_examples) < prereg["failure_case_sample"]:
                    text = d.x.decode("utf-8", "replace")
                    spans = [m_.group(0) for m_ in _SENT_RE.finditer(text)
                             if m_.group(0).strip()]
                    for sp, st in zip(spans, sents):
                        if (not st.supported and len(unsupported_examples)
                                < prereg["failure_case_sample"]):
                            unsupported_examples.append(
                                {"kind": "unsupported_sentence", "key": d.key,
                                 "split": split_name, "reason": st.reason,
                                 "text": sp.strip()[:300]})
            else:
                reasons["non_utf8_document"] += 1
    roundtrip = {"documents": len(train) + len(test),
                 "mode_A_fallback": mode_counts["mode_A"],
                 "mode_B_predictive": mode_counts["mode_B"],
                 "by_arm": {a: {"smode": arms[a][0], "cset": arms[a][1],
                                "exact": rt[a]["ok"], "failed": rt[a]["fail"]}
                            for a in names},
                 "all_exact": all(rt[a]["fail"] == 0 for a in names),
                 "checks": ["length", "crc32", "byte equality",
                            "state replay validation", "canonical re-encoding",
                            "surface choice range/default checks"]}
    _dump(out, "roundtrip_results.json", roundtrip)

    def E(d, a=None) -> codec.Encoded:
        return enc[id(d)][a or names[0]]

    def P(d):
        return enc[id(d)][names[0]]

    # ------------------------------------------------ coverage
    def cov(part):
        nd = len(part)
        docs_b = docs_full = 0
        s_tot = s_sup = b_tot = b_sup = 0
        ents = vers = ana = 0
        for d in part:
            e = P(d)
            b_tot += len(d.x)
            if e.parse is None:
                continue
            st = e.parse.sentences
            s_tot += len(st)
            s_sup += sum(x.supported for x in st)
            if e.mode == codec.MODE_B:
                docs_b += 1
                b_sup += sum(x.nbytes for x in st if x.supported)
                ents += e.parse.n_entities
                vers += e.parse.n_versions
                ana += e.parse.n_anaphora
                docs_full += all(x.supported for x in st)
        return {"documents": nd,
                "doc_mode_B_fraction": docs_b / max(1, nd),
                "doc_fully_covered_fraction": docs_full / max(1, nd),
                "sentence_coverage": s_sup / max(1, s_tot),
                "byte_weighted_coverage": b_sup / max(1, b_tot),
                "sentences": s_tot,
                "mean_entities_per_mode_B_doc": ents / max(1, docs_b),
                "mean_state_versions_per_mode_B_doc": vers / max(1, docs_b),
                "mean_anaphora_resolved_per_mode_B_doc": ana / max(1, docs_b)}

    coverage = {"train": cov(train), "test": cov(test), "whole": cov(train + test),
                "unsupported_reasons": dict(reasons.most_common()),
                "state_tracking_accuracy": {
                    "status": "UNMEASURED",
                    "why": "requires a gold-annotated entity-state sample; "
                           "parser statistics above are not accuracy"}}
    _dump(out, "parser_coverage.json", coverage)
    bcov = coverage["test"]["byte_weighted_coverage"]

    # ------------------------------------------------ coding rates
    covered = {id(d) for d in test if P(d).mode == codec.MODE_B
               and all(x.supported for x in P(d).parse.sentences)}
    nbt = max(1, sum(len(d.x) for d in test))
    sym_cache: dict[tuple, list] = {}

    def s_syms(d, a):
        key = ("s", id(d), arms[a][0])
        if key not in sym_cache:
            sym_cache[key] = serial.symbols(E(d, a).s, arms[a][0])
        return sym_cache[key]

    def c_syms(d, a):
        e = E(d, a)
        key = ("c", id(d), arms[a][1])
        if key not in sym_cache:
            sym_cache[key] = (surface.symbols(e.parse.instrs, e.c, arms[a][1])
                              if e.mode == codec.MODE_B else [])
        return sym_cache[key]

    families = {
        "byte": (metrics.WBContextModel,
                 lambda d: metrics.baseline_stream(d.x),
                 lambda d, a: metrics.hybrid_stream(E(d, a).mode, E(d, a).s,
                                                    E(d, a).r, E(d, a).c)),
        "symbol": (metrics.WBSymbolModel,
                   lambda d: metrics.baseline_symbols(d.x),
                   lambda d, a: metrics.hybrid_symbols(E(d, a).mode, s_syms(d, a),
                                                       E(d, a).r, c_syms(d, a))),
    }
    coding = {
        "family_note": ("Within each family: same model class, order and train "
                        "split for every arm and the baseline; held-out test "
                        "costs; model parameters excluded symmetrically. "
                        "L_hybrid = L(S) + L(C|S) + L(R|S,C)."),
        "families": {
            "byte": {"desc": "Witten-Bell order-k over bytes",
                     "hybrid": "MODE + S + SEP | C + CSEP | R + EOR"},
            "symbol": {"desc": "Witten-Bell order-k over model-facing symbols",
                       "hybrid": "<MODE> + S symbols + <SEP> | C symbols + <CSEP> "
                                 "| R bytes + <EOR>; unseen symbols pay ESC + 8 "
                                 "bits/char"}},
        "arms": {a: {"smode": sm, "cset": cs,
                     "encoding": serial.MODES[sm][0], "state": serial.MODES[sm][1],
                     "checkpoint_k": serial.MODES[sm][2],
                     "surface_fields": list(surface.CSETS[cs][0]),
                     "permuted_ids": surface.CSETS[cs][1]}
                 for a, (sm, cs) in arms.items()},
        "results": {}}
    per_doc: dict[str, dict[str, np.ndarray]] = {}
    for fam, (Model, bstream, hstream) in families.items():
        coding["results"][fam] = {a: {} for a in names}
        for k in prereg["coding_orders"]:
            mb = Model(k).fit(bstream(d) for d in train)
            LX = np.array([mb.costs(bstream(d)).sum() for d in test])
            NB = np.array([len(d.x) for d in test], dtype=float)
            COV = np.array([1.0 if id(d) in covered else 0.0 for d in test])
            if k == kord:
                per_doc[fam] = {"LX": LX, "NB": NB, "COV": COV}
            for a in names:
                mh = Model(k).fit(hstream(d, a)[0] for d in train)
                LS, LC, LR = [], [], []
                for d in test:
                    hs, cs_, cc_ = hstream(d, a)
                    c = mh.costs(hs)
                    LS.append(c[:cs_].sum())
                    LC.append(c[cs_:cc_].sum())
                    LR.append(c[cc_:].sum())
                LS, LC, LR = np.array(LS), np.array(LC), np.array(LR)
                tx, ts, tc, tr, nb = LX.sum(), LS.sum(), LC.sum(), LR.sum(), NB.sum()
                th = ts + tc + tr
                msk = COV > 0
                coding["results"][fam][a][f"order_{k}"] = {
                    "test_source_bytes": int(nb),
                    "L_X_bits": float(tx), "L_S_bits": float(ts),
                    "L_C_given_S_bits": float(tc), "L_R_given_SC_bits": float(tr),
                    "baseline_bits_per_source_byte": float(tx / max(1, nb)),
                    "S_bits_per_source_byte": float(ts / max(1, nb)),
                    "C_bits_per_source_byte": float(tc / max(1, nb)),
                    "R_bits_per_source_byte": float(tr / max(1, nb)),
                    "hybrid_bits_per_source_byte": float(th / max(1, nb)),
                    "gamma_no_shared": float(th / max(1e-9, tx)),
                    "rho_R": float(tr / max(1e-9, th)),
                    "covered_subset": {
                        "docs": int(msk.sum()),
                        "gamma_no_shared": (float((LS[msk].sum() + LC[msk].sum()
                                                   + LR[msk].sum()) / LX[msk].sum())
                                            if msk.any() else None)}}
                if k == kord:
                    per_doc[fam][f"LS_{a}"] = LS
                    per_doc[fam][f"LC_{a}"] = LC
                    per_doc[fam][f"LR_{a}"] = LR
    coding["lzma_xz_preset9e"] = {}
    lx = metrics.lzma_bits(metrics.baseline_stream(d.x) for d in test)
    for a in names:
        lh = metrics.lzma_bits(metrics.hybrid_stream(E(d, a).mode, E(d, a).s,
                                                     E(d, a).r, E(d, a).c)[0]
                               for d in test)
        coding["lzma_xz_preset9e"][a] = {"L_X_bits": lx, "L_hybrid_bits": lh,
                                         "gamma_no_shared": lh / max(1, lx)}

    # ------------------------------------------------ coder-order sweep
    # Preregistered sensitivity analysis: how gamma moves with predictor
    # context. Order `primary_order` remains the only decision input.
    s_orders = prereg.get("sensitivity_orders", [])
    s_arms = [a for a in prereg.get("sensitivity_arms", []) if a in arms]
    sweep = {"orders": s_orders, "arms": s_arms, "primary_order": kord,
             "note": ("sensitivity analysis only; the decision uses the "
                      "preregistered primary order"),
             "by_family": {}}
    if s_orders and s_arms:
        for fam, (Model, bstream, hstream) in families.items():
            sweep["by_family"][fam] = {}
            for k in s_orders:
                mb = Model(k).fit(bstream(d) for d in train)
                cols = {"LX": np.array([mb.costs(bstream(d)).sum() for d in test])}
                row = {}
                for a in s_arms:
                    mh = Model(k).fit(hstream(d, a)[0] for d in train)
                    cols[a] = np.array([mh.costs(hstream(d, a)[0]).sum()
                                        for d in test])
                fns = {f"{a}": (lambda s, a=a: s[a] / max(1e-9, s["LX"]))
                       for a in s_arms}
                ref = "C0" if "C0" in s_arms else None
                if ref:
                    fns.update({f"{a}-{ref}": (lambda s, a=a: (s[a] - s[ref])
                                               / max(1e-9, s["LX"]))
                                for a in s_arms if a != ref})
                bt = metrics.bootstrap(cols, fns, prereg["bootstrap_resamples"], seed)
                for a in s_arms:
                    row[a] = {"gamma": bt[a]["estimate"],
                              "gamma_ci95": bt[a]["ci95"],
                              "hybrid_beats_baseline": bt[a]["ci95"][1] < 1.0}
                    if ref and a != ref:
                        row[a]["delta_gamma_vs_C0"] = bt[f"{a}-{ref}"]["estimate"]
                        row[a]["delta_gamma_vs_C0_ci95"] = bt[f"{a}-{ref}"]["ci95"]
                row["baseline_bits_per_source_byte"] = float(cols["LX"].sum() / nbt)
                sweep["by_family"][fam][f"order_{k}"] = row
            for a in s_arms:
                gs = [sweep["by_family"][fam][f"order_{k}"][a]["gamma"] for k in s_orders]
                wins = [k for k in s_orders
                        if sweep["by_family"][fam][f"order_{k}"][a]["hybrid_beats_baseline"]]
                sweep["by_family"][fam].setdefault("summary", {})[a] = {
                    "orders_where_hybrid_beats_baseline_ci": wins,
                    "gamma_monotone_nondecreasing_in_order":
                        all(x <= y + 1e-12 for x, y in zip(gs, gs[1:])),
                    "spearman_gamma_vs_order": float(
                        np.corrcoef(np.argsort(np.argsort(s_orders)),
                                    np.argsort(np.argsort(gs)))[0, 1])
                        if len(s_orders) > 2 else None}
        fig = figures.order_sweep(sweep, out, kord)
        sweep["figure"] = fig
    coding["order_sweep"] = sweep

    # ------------------------------------------------ arm decomposition
    def parts(fam, a):
        r = coding["results"][fam][a][pk]
        return (r["S_bits_per_source_byte"], r["C_bits_per_source_byte"],
                r["R_bits_per_source_byte"], r["gamma_no_shared"])

    contrasts = [(a, b, label) for a, b, label in prereg.get("arm_contrasts", [])
                 if a in arms and b in arms]
    decomposition = {"primary_order": kord,
                     "primary_contrast": prereg.get("primary_contrast"),
                     "unit": "bits per source byte; delta_gamma = change in "
                             "hybrid/baseline (negative = improvement)",
                     "by_family": {}}
    for fam in families:
        rows = []
        for a, b, label in contrasts:
            sa, ca, ra, ga = parts(fam, a)
            sb, cb, rb, gb = parts(fam, b)
            rows.append({"contrast": f"{a} - {b}", "meaning": label,
                         "delta_S_bpb": sa - sb, "delta_C_bpb": ca - cb,
                         "delta_R_bpb": ra - rb,
                         "delta_total_bpb": (sa + ca + ra) - (sb + cb + rb),
                         "delta_gamma": ga - gb})
        decomposition["by_family"][fam] = rows

    # ------------------------------------------------ shared artifacts (MDL)
    blob = lexicon.table_bytes() + b"".join((PKG / f).read_bytes()
                                            for f in COMPILER_SOURCES)
    shared_bits = 8 * len(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME))
    target = prereg["target_corpus_bytes_for_amortization"]

    def gamma_at(fam, a, nbytes):
        r = coding["results"][fam][a][pk]
        return r["gamma_no_shared"] + shared_bits / max(
            1e-9, r["baseline_bits_per_source_byte"] * nbytes)

    shared = {"components": ["rule tables (lexicon.table_bytes)"] +
              [f"src/hsir/{f}" for f in COMPILER_SOURCES],
              "raw_bytes": len(blob), "lzma_bits": shared_bits,
              "rule_table_sha256": _sha(lexicon.table_bytes()),
              "rules_version": lexicon.RULES_VERSION,
              "surface_schema_version": surface.SURFACE_SCHEMA_VERSION,
              "corpus_dependent_dictionaries": "none (all rules and surface "
                                               "value lists hand-written, frozen "
                                               "before evaluation)",
              "note": "one compiler (including the surface layer) serves every "
                      "arm, so L_shared is common to all arms",
              "gamma_with_shared": {
                  fam: {a: {"at_test_split": coding["results"][fam][a][pk]["gamma_no_shared"]
                            + shared_bits / max(1e-9, coding["results"][fam][a][pk]["L_X_bits"]),
                            "at_1MB": gamma_at(fam, a, 1e6),
                            "at_100MB": gamma_at(fam, a, 1e8),
                            "at_1GB": gamma_at(fam, a, 1e9),
                            "at_prereg_target": gamma_at(fam, a, target)}
                        for a in names}
                  for fam in families},
              "prereg_target_bytes": target}
    _dump(out, "shared_artifact_costs.json", shared)

    # ------------------------------------------------ residual + surface usage
    surface_usage = {}
    for cs in csets:
        if cs == "C0":
            continue
        arm = next(a for a in names if arms[a][1] == cs)
        agg = {}
        for d in test:
            for f, v in E(d, arm).cstats.items():
                g = agg.setdefault(f, Counter())
                g.update(v)
        surface_usage[cs] = {
            "fields": {f: {**dict(g),
                           "emit_rate": g["emitted"] / max(1, g["slots"]),
                           "fallback_rate_unrepresentable":
                               g["unrepresentable"] / max(1, g["slots"] - g["unobserved"]),
                           "unobserved_rate": g["unobserved"] / max(1, g["slots"])}
                       for f, g in agg.items()},
            "C_bytes_per_source_byte": sum(len(E(d, arm).c) for d in test) / nbt,
            "R_bytes_per_source_byte": sum(len(E(d, arm).r) for d in test) / nbt}
    residual = {
        "primary_order": kord,
        "R_bytes_per_source_byte": {a: sum(len(E(d, a).r) for d in test) / nbt
                                    for a in names},
        "C_bytes_per_source_byte": {a: sum(len(E(d, a).c) for d in test) / nbt
                                    for a in names},
        "S_bytes_per_source_byte": {a: sum(len(E(d, a).s) for d in test) / nbt
                                    for a in names},
        "S_symbols_per_source_byte": {a: sum(len(s_syms(d, a)) for d in test) / nbt
                                      for a in names},
        "container_framing_bytes_per_source_byte": {
            a: sum(codec.framing_bytes(d.x, E(d, a)) for d in test) / nbt
            for a in names},
        "by_family": {fam: {a: {
            "rho_R_residual_share_of_hybrid": coding["results"][fam][a][pk]["rho_R"],
            "residual_cost_relative_to_baseline":
                coding["results"][fam][a][pk]["L_R_given_SC_bits"]
                / max(1e-9, coding["results"][fam][a][pk]["L_X_bits"]),
            "surface_cost_relative_to_baseline":
                coding["results"][fam][a][pk]["L_C_given_S_bits"]
                / max(1e-9, coding["results"][fam][a][pk]["L_X_bits"]),
            "semantic_cost_relative_to_baseline":
                coding["results"][fam][a][pk]["L_S_bits"]
                / max(1e-9, coding["results"][fam][a][pk]["L_X_bits"])}
            for a in names} for fam in families},
        "surface_choice_usage_test_split": surface_usage,
        "arm_decomposition": decomposition,
        "conditioning_note": ("R is an edit script against G(S, C); it changes "
                              "with the surface set but not with the S "
                              "serialization mode."),
        "threshold_rho_R": thr["rho_R_diagnostic_max"]}
    _dump(out, "residual_allocation.json", residual)

    # ------------------------------------------------ rank-frequency
    def gather(part):
        ops, opnds, pairs = [], [], []
        z3 = {a: [] for a in names}
        for d in part:
            e = P(d)
            for a in names:
                ea = E(d, a)
                if ea.mode != codec.MODE_B:
                    z3[a].extend(["<MODE_A>"] + [f"b{c:02x}" for c in ea.r])
                else:
                    z3[a].extend(s_syms(d, a) + ["<SEP>"] + c_syms(d, a)
                                 + ["<CSEP>"] + [f"b{c:02x}" for c in ea.r])
            if e.mode != codec.MODE_B:
                continue
            o, a_, p = _analyse(e.parse.instrs)
            ops += o
            opnds += a_
            pairs += p
        return ops, opnds, pairs, z3

    otr, atr, ptr, ztr = gather(train)
    ote, ate, pte, zte = gather(test)
    cats = sorted({c for c, _ in atr} | {c for c, _ in ate})
    rank = {"H_Z1_operators": metrics.rank_frequency(otr, ote),
            "H_Z2_operands": {c: metrics.rank_frequency(
                [t for k_, t in atr if k_ == c], [t for k_, t in ate if k_ == c])
                for c in cats},
            "H_Z3_serialized_symbols": {a: metrics.rank_frequency(ztr[a], zte[a])
                                        for a in names},
            "conditional_entropy_concept_given_action":
                metrics.conditional_entropy(ptr, pte),
            "notes": ["H_Z1/H_Z2 are computed on the canonical (explicit) "
                      "instruction list and do not depend on the arm",
                      "entity_ref operands are document-local IDs; their rank "
                      "distribution reflects numbering, not language",
                      "compare held-out bits/token across models rather than "
                      "reading a fitted exponent alone"]}
    _dump(out, "rank_frequency.json", rank)

    # ------------------------------------------------ nonsemantic control
    def hyb_symbols(part, a):
        n = 0
        for d in part:
            n += len(s_syms(d, a)) + len(c_syms(d, a)) + len(E(d, a).r) + 4
        return n / max(1, sum(len(d.x) for d in part))

    per_arm_sym = {a: {"train": hyb_symbols(train, a), "test": hyb_symbols(test, a)}
                   for a in names}
    tgt_arm = min(names, key=lambda a: per_arm_sym[a]["train"])
    tgt = per_arm_sym[tgt_arm]["train"]
    bpe = BPE()
    if tgt < 1.0:
        fit = bpe.fit([d.x for d in train], tgt)
    else:
        fit = {"merges": 0, "note": "every arm expands beyond 1 symbol/byte; "
                                    "raw bytes are the closest reversible control"}
    ids_n = sum(len(bpe.encode(d.x)) for d in test)
    control = {"hybrid_symbols_per_byte": per_arm_sym,
               "length_matched_to_arm": tgt_arm,
               "bpe_fit": fit,
               "bpe_symbols_per_byte_test": ids_n / nbt,
               "bpe_reversible_on_test": all(bpe.decode(bpe.encode(d.x)) == d.x
                                             for d in test)}
    coding["nonsemantic_bpe_control"] = control
    coding["arm_decomposition"] = decomposition
    _dump(out, "coding_rates.json", coding)

    # ------------------------------------------------ bootstrap (paired)
    def tot(s, a):
        return s[f"LS_{a}"] + s[f"LC_{a}"] + s[f"LR_{a}"]

    boot = {"_note": ("paired document-level percentile bootstrap on the test "
                      "split at the primary order; every arm is resampled with "
                      "the same document indices, so contrasts are paired; "
                      "gamma excludes L_shared")}
    for fam in families:
        fns = {}
        for a in names:
            fns[f"{a}.gamma_no_shared"] = (lambda s, a=a: tot(s, a) / max(1e-9, s["LX"]))
            fns[f"{a}.rho_R"] = (lambda s, a=a: s[f"LR_{a}"] / max(1e-9, tot(s, a)))
        for a, b, _ in contrasts:
            fns[f"delta_gamma[{a}-{b}]"] = (
                lambda s, a=a, b=b: (tot(s, a) - tot(s, b)) / max(1e-9, s["LX"]))
        fns["baseline_bits_per_byte"] = lambda s: s["LX"] / max(1, s["NB"])
        fns["covered_doc_fraction"] = lambda s: s["COV"] / max(1, len(test))
        boot[fam] = metrics.bootstrap(per_doc[fam], fns,
                                      prereg["bootstrap_resamples"], seed)
    _dump(out, "bootstrap_intervals.json", boot)

    # ------------------------------------------------ decision
    by_arm = {}
    for a in names:
        fams = {}
        for fam in families:
            r = coding["results"][fam][a][pk]
            inputs = {"rho_R": r["rho_R"],
                      "gamma_whole_no_shared": r["gamma_no_shared"],
                      "gamma_whole_ci95": boot[fam][f"{a}.gamma_no_shared"]["ci95"],
                      "gamma_at_prereg_target_with_shared":
                          shared["gamma_with_shared"][fam][a]["at_prereg_target"],
                      "gamma_covered_subset": r["covered_subset"]["gamma_no_shared"]}
            fams[fam] = {**_decide(inputs, bcov, thr), "inputs": inputs}
        worst = max((v["decision"] for v in fams.values()), key=SEVERITY.__getitem__)
        by_arm[a] = {"decision": worst, "by_family": fams}
    if not roundtrip["all_exact"]:
        decision = would_be = "INVALID"
        reasons_d = ["round-trip failures: implementation bug, no measurement "
                     "is trustworthy"]
        eligible = []
    else:
        would_be = min((v["decision"] for v in by_arm.values()),
                       key=SEVERITY.__getitem__)
        eligible = [a for a, v in by_arm.items() if v["decision"] == "GO"]
        best = [a for a, v in by_arm.items() if v["decision"] == would_be]
        reasons_d = list(dict.fromkeys(
            f"[{a}/{fam}] {r}" for a in best
            for fam, fv in by_arm[a]["by_family"].items() for r in fv["reasons"]))
        decision = "PENDING_REAL_DATA" if synthetic_run else would_be
    feas = {"decision": decision,
            "diagnostic_decision_ignoring_synthetic_flag": would_be,
            "combination_rule": ("per arm: most severe across coding families "
                                 "(byte, symbol); overall: least severe across "
                                 "preregistered arms"),
            "selection_caveat": (f"overall decision selects the best of "
                                 f"{len(names)} preregistered arms on the same "
                                 "test split; confirm an eligible arm on a fresh "
                                 "split before Phase III"),
            "eligible_arms_for_phase_III": eligible,
            "by_arm": by_arm,
            "reasons": reasons_d,
            "byte_coverage_test": bcov,
            "thresholds": thr,
            "not_assessed_in_phase_II": [
                "semantic usefulness (requires Phase III training)",
                "state-tracking accuracy (requires gold annotations)",
                "whether explicit state or surface choices help a learner "
                "(arm contrasts are coding-cost comparisons only)",
                "whether a stronger conditional coder would price C differently "
                "(order-k context models cannot exploit long-range dependencies)"],
            "model_training_permitted": decision == "GO"}
    _dump(out, "feasibility_decision.json", feas)

    # ------------------------------------------------ failures + manifest
    with open(out / "failure_cases.jsonl", "w") as f:
        for rec in failures + unsupported_examples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {
        "hsir_phase": "II",
        "protocol_version": "0.3.3",
        "created_unix": int(time.time()),
        "runtime_seconds": round(time.time() - t0, 2),
        "dataset": source_desc,
        "synthetic": synthetic_run,
        "compiler": {"format_version": codec.FORMAT_VERSION,
                     "rules_version": lexicon.RULES_VERSION,
                     "surface_schema_version": surface.SURFACE_SCHEMA_VERSION,
                     "arms": {a: list(v) for a, v in arms.items()},
                     "rule_table_sha256": _sha(lexicon.table_bytes()),
                     "source_sha256": {f: _sha((PKG / f).read_bytes())
                                       for f in COMPILER_SOURCES}},
        "coding_models": {"families": list(families),
                          "orders": prereg["coding_orders"],
                          "primary_order": kord},
        "prereg_sha256": _file_sha(str(prereg_path)),
        "prereg": prereg,
        "seed": seed,
        "environment": {"python": sys.version.split()[0],
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()},
        "files": ["manifest.json", "dataset_audit.json", "parser_coverage.json",
                  "roundtrip_results.json", "coding_rates.json",
                  "residual_allocation.json", "rank_frequency.json",
                  "bootstrap_intervals.json", "shared_artifact_costs.json",
                  "failure_cases.jsonl", "feasibility_decision.json"]}
    _dump(out, "manifest.json", manifest)
    return feas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="RecipeNLG CSV or JSONL with 'directions'")
    src.add_argument("--synthetic", type=int, metavar="N",
                     help="N synthetic docs (unit-test corpus; decision forced "
                          "to PENDING_REAL_DATA)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prereg", default=str(PREREG_PATH))
    a = ap.parse_args(argv)
    prereg = json.loads(Path(a.prereg).read_text())
    if a.synthetic:
        texts = synthetic.corpus(a.synthetic, prereg["seed"])
        docs = [ds.Doc(f"synthetic-{i}", t.encode()) for i, t in enumerate(texts)]
        load_audit = {"rows": len(docs), "loaded": len(docs)}
        desc = {"kind": "synthetic", "n": a.synthetic, "generator_seed": prereg["seed"]}
    else:
        docs, load_audit = ds.load(a.input, a.limit)
        desc = {"kind": "file", "path": os.path.abspath(a.input),
                "sha256": _file_sha(a.input), "limit": a.limit}
    is_synth = bool(a.synthetic) or load_audit.get("synthetic_records", 0) > 0
    feas = run(docs, load_audit, Path(a.out), prereg, desc, is_synth,
               Path(a.prereg))
    print(json.dumps({"decision": feas["decision"],
                      "diagnostic": feas["diagnostic_decision_ignoring_synthetic_flag"],
                      "eligible_arms": feas["eligible_arms_for_phase_III"],
                      "by_arm": {a: v["decision"] for a, v in feas["by_arm"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
