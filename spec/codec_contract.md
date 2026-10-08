# Codec and Phase II contract (protocol v0.3.3)

## Invariant

For every byte string `x`: `decode(encode(x)) == x`. Verified by length,
CRC-32 and byte equality. No Unicode normalization anywhere.

## Container (format v3)

```
"HSIR" | 3 | mode:u8 | smode:u8 | cset:u8 | varint |x| |S| |C| |R| | S | C | R | crc32(x):u32be
```

`smode` indexes `serial.MODE_IDS`; `cset` indexes `surface.CSET_IDS`
(C0, C1, C2, C3, C4, C4P); both append-only. Format v2 containers (no cset,
no C) still decode as C0. `R` depends on the surface set but not on the S
serialization mode. See `surface_schema.md`.

* **Mode A** (raw fallback): `S` and `C` empty, `R = x`. Chosen iff `x` is not strict
  UTF-8 or contains no supported sentence. Never chosen on size grounds.
* **Mode B** (predictive): `R = diff(G(S, C), x)`; `R` is conditioned on
  `S` and `C` by construction.

## Residual

ASCII-framed edit ops against `ŷ = realize(S)`:
`R<skip>,<del>,<ins>:<bytes>` · `D<skip>,<del>;` · `I<skip>,<ins>:<bytes>`.
Payloads are length-delimited raw bytes. Equal runs < 6 bytes between edits
are coalesced (codec parameter, data-independent).

## Measured quantities

* `L_hybrid = L(S) + L(C|S) + L(R|S,C)`, every term charged.
* `ρ_R = L(R|S,C) / L_hybrid` — residual share of the hybrid code.
* `γ = (L_hybrid + L_shared) / L(X)` — total coding efficiency.
* Arm contrasts report ΔL(S), ΔL(C), ΔL(R) and Δγ separately, so a smaller
  residual bought with a larger C channel is visible.

All `L` are held-out test bits under one model family per comparison,
trained on the train split. Two families, both reported:

* **byte**: Witten–Bell order-k over bytes.
* **symbol**: Witten–Bell order-k over model-facing symbols (S tokens + R
  bytes vs source bytes); unseen symbols pay escape + 8 bits/char.

An **arm** is a (serialization mode, surface set) pair, preregistered in
`prereg.json`. Per arm, the decision takes the **most severe** outcome
across families. The overall decision is the **least severe** across the
preregistered arms, with a selection caveat: an eligible arm must be
confirmed on a fresh split before Phase III. Arm contrasts get paired
document-level bootstrap intervals.
`L_shared` = lzma size of rule tables + compiler source, amortized at
1 MB / 100 MB / 1 GB and at the preregistered target size.

## Output files

| File | Content |
|---|---|
| `manifest.json` | dataset hash, compiler hashes, prereg hash, seed, env |
| `dataset_audit.json` | load counts, exact/near-dup removal, split sizes, fields used |
| `parser_coverage.json` | doc / sentence / byte-weighted coverage, failure reasons |
| `roundtrip_results.json` | exact round trips over every document |
| `coding_rates.json` | per family × order: L(X), L(S), L(R|S), γ, ρ_R; lzma; BPE control |
| `residual_allocation.json` | ρ_R and cost ratios per family; byte-length ratios |
| `rank_frequency.json` | H_Z1 operators, H_Z2 operands by category, H_Z3 symbols, H(concept|action) |
| `bootstrap_intervals.json` | document-level 95% CIs per family |
| `shared_artifact_costs.json` | L_shared and amortized γ |
| `failure_cases.jsonl` | round-trip failures and a sample of unsupported sentences |
| `feasibility_decision.json` | GO / REVISE / STOP / INVALID / PENDING_REAL_DATA |

Optional figure: `order_sweep.png` (γ vs context-model order, written when
matplotlib is installed). The underlying numbers, with paired bootstrap
intervals, are in `coding_rates.json` → `order_sweep`. The sweep
(`sensitivity_orders`, `sensitivity_arms` in `prereg.json`) is a sensitivity
analysis; only `primary_order` feeds the decision.

Thresholds live in `spec/prereg.json`; its SHA-256 is in the manifest.
Synthetic runs are forced to `PENDING_REAL_DATA`.
