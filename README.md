# H-SIR Phase II — codec and feasibility harness (protocol v0.3.3)

[![tests](https://github.com/danindiana/h-sir/actions/workflows/tests.yml/badge.svg)](https://github.com/danindiana/h-sir/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![tests](https://img.shields.io/badge/tests-399-2ea44f)
![round trip](https://img.shields.io/badge/round%20trip-byte--exact-2ea44f)
![protocol](https://img.shields.io/badge/protocol-v0.3.3%20frozen-6f42c1)
![phase II](https://img.shields.io/badge/phase%20II-pending%20real%20data-d29922)
![synthetic diagnostic](https://img.shields.io/badge/synthetic%20diagnostic-REVISE-d29922)
![license](https://img.shields.io/badge/license-not%20yet%20chosen-lightgrey)
[![diagrams](https://img.shields.io/badge/diagrams-graphviz-58a6ff)](docs/diagrams/)

Reversible semantic-instruction codec for recipe text, plus the measurement
harness that decides whether H-SIR is worth training a transformer on.
**No model training happens here.** Training is permitted only after a
real-data report returns `GO`.

```
E(x) = (S, C, R)   S = semantic instruction stream (rule-based, versioned entities)
                   C = sparse categorical surface choices (separators, articles, ...)
                   R = byte-exact edit script from G(S, C) to x
D(E(x)) == x       for every byte string, verified by length + CRC + equality
```

![H-SIR codec pipeline](docs/diagrams/01_codec_pipeline.svg)

## Why this exists

Language models learn from text whose word frequencies follow a heavy-tailed,
roughly Zipfian pattern, and much of what they learn is surface habit: which
article, which synonym, where the line breaks fall. The question behind H-SIR
is whether rewriting text as compact, compositional *instructions* (in the
spirit of array languages like APL, BQN and Uiua) would let a model spend its
capacity on structure instead.

That idea is easy to over-claim. A rewrite can look "more efficient" simply
because it throws information away, because a parser quietly imports
knowledge from elsewhere, or because the token counts are not comparable. This
repository exists to answer a narrower question first, with the rules fixed in
advance: **is the representation even economical, before any model is
trained on it?**

## In plain terms

Take a recipe line: *"Combine flour, sugar and the butter in a bowl."*

H-SIR splits it into three parts:

1. **Meaning** — a short instruction: *mix flour, sugar and butter into a new
   mixture, inside a bowl.* It also tracks each ingredient as it changes
   ("the mixture" after mixing is a new state of the same thing).
2. **Style notes** — small multiple-choice facts about the wording: "Combine"
   rather than "Mix", no article before *flour*, no comma before *and*, a
   line break before the next sentence.
3. **Patches** — anything the rules could not express, copied exactly.

Recombining the three always reproduces the original text **byte for byte**.
Nothing is ever lost, so any saving has to be real.

The test is then simple to state: does a predictor need **fewer bits** to
describe *meaning + style notes + patches* than to describe the original text?
The ratio is called **γ** (hybrid bits ÷ original bits). Below 1 means H-SIR is
cheaper; above 1 means it costs more.

So far, on made-up recipes built to suit the parser, the answer at the
standard setting is **no**: γ ≈ 1.3. It is **yes** only for predictors with
very short memory (γ ≈ 0.6 when the predictor sees one previous symbol). Real
recipes are the next test.

## Summary

| | |
|---|---|
| **What is built** | A lossless codec `E(x) = (S, C, R)` for recipe text; 7 ways of writing S, 6 surface-choice sets for C; a byte-exact residual R; a preregistered harness that scores 13 combinations ("arms") with matched predictors and paired confidence intervals |
| **Correctness** | 399 tests and stress runs of up to 179,000 round trips, all byte-exact, including random binary input and malformed streams |
| **Best arm so far** | `C4` (implicit state + full surface layer): γ = 1.30 [1.28, 1.32], symbol predictor, order 3, synthetic data |
| **Phase II status** | Synthetic diagnostic **REVISE**; decision **pending real data** (RecipeNLG run not yet done) |
| **Training** | None. Not permitted until a real-data report returns `GO`, or a documented pivot to the inductive-bias track (`spec/phase3_prediction.md`) |

## Analysis (synthetic evidence only)

All numbers below come from `examples/phase2_synthetic/` (2,000 generated
recipes, order-3 context models, 95% paired bootstrap intervals). The
generator shares the parser's vocabulary, so these results are *favourable*
to H-SIR; real text can only make coverage worse.

**Where the bits go.** With the symbol predictor the original text costs 0.77
bits per byte. The best arm spends 0.58 on meaning, 0.18 on style notes and
0.24 on patches. Meaning plus style notes alone already reach 0.76, so even a
free residual would leave γ ≈ 0.99 (≈ 1.11 with the byte predictor).
Further residual tuning cannot reach `GO`; the semantic stream itself would
have to get cheaper.

**The surface layer works.** Replacing byte patches with style notes cuts the
patch cost from 0.65 to 0.24 bits per byte and lowers γ by 0.30
[−0.32, −0.28]. Each group of notes saves at least twice what it costs. A
control that scrambles the note labels changes nothing (as it should).

**Explicit state is expensive to spell out.** Writing every entity version
explicitly costs +0.20 in γ over letting the decoder recompute it, and
verbose ASCII spelling adds another +0.12. That information is fully
derivable, so its cost measures how hard it is for a short-memory predictor
to *track* state, not how much information it carries.

**The advantage depends on predictor strength.** H-SIR beats the original
text for weak predictors (orders 0–2, γ as low as 0.59) and loses as the
predictor sees more context (rank correlation of γ with order ≈ 0.89). The
style-note gain itself stays roughly constant across orders.

**What this does not show.** Nothing here is evidence about transformers: an
order-k context model is not a neural network. Coverage on real recipes is
unknown. And a representation that compresses poorly may still be a useful
*inductive bias* for small models, which is the preregistered Phase III
hypothesis (capacity substitution).

## Rationale

| Decision | Why |
|---|---|
| Lossless by construction (`D(E(x)) = x`) | No gain can come from silently dropping information |
| Rule-based parser, no neural parser | A pretrained parser would smuggle outside knowledge into the representation |
| Only the `directions` field is read | Titles and ingredient lists would be privileged information the baseline never sees |
| Whole-corpus accounting, fallback included | A parser that handles only easy sentences must not look good; covered-subset results are reported separately |
| Same predictor class for both sides, two families | Measures the representation, not the quality of one codec; the worse of the two decides |
| Rule tables and compiler code charged as shared cost (MDL) | Knowledge baked into the rules is not free |
| Exact and near-duplicate removal before a document-level split | Prevents the test set from leaking into training |
| Preregistered thresholds, frozen files, synthetic runs forced to `PENDING_REAL_DATA` | The hypothesis cannot be adjusted after seeing results |
| Paired bootstrap, permuted-label and length-matched controls | Differences between arms are tested on the same documents, and spurious gains have a place to show up |
| Recipes as the domain | Compositional actions, ingredients that change state, and real wording variation, in a domain small enough for a rule-based parser |

## Setup (Ubuntu 24.04, bash/zsh)

```bash
cd h-sir
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test,figures]'
pytest -q                      # 399 tests: round trips, modes, surface sets, residual, metrics, report, freeze
```

## Run Phase II

```bash
python experiments/freeze.py --verify            # v0.3.3 is frozen; must print "freeze intact"

# Real data: RecipeNLG CSV (only the `directions` column is read)
hsir-phase2 --input /path/to/RecipeNLG_dataset.csv --limit 50000 --out phase2/

# Synthetic smoke run (decision forced to PENDING_REAL_DATA)
hsir-phase2 --synthetic 2000 --out phase2_synth/

python experiments/entropy_report.py phase2/     # one-screen summary
python experiments/rank_frequency.py phase2/     # Zipf vs stretched-exp, H(concept|action)
```

Runtime is pure Python: roughly 50 s per 1,000 short recipes (13 arms × two
coding families × three orders, plus the order-0…6 sensitivity sweep on four
arms). Start with `--limit 20000`; the full 2.2 M-row corpus will
take hours and is not needed for a feasibility gate.

## Diagrams

Graphviz sources and renders live in [`docs/diagrams/`](docs/diagrams/)
(`bash docs/diagrams/render.sh` regenerates them).

| | |
|---|---|
| [Codec pipeline](docs/diagrams/01_codec_pipeline.svg) | [Container format v3](docs/diagrams/02_container_format_v3.svg) |
| [Entity state machine](docs/diagrams/03_entity_state_machine.svg) | [Serialization modes](docs/diagrams/04_serialization_modes.svg) |
| [Surface layer](docs/diagrams/05_surface_layer.svg) | [Phase II harness](docs/diagrams/06_phase2_harness.svg) |
| [Decision logic](docs/diagrams/07_decision_logic.svg) | [Research roadmap](docs/diagrams/08_research_roadmap.svg) |

![Phase II harness](docs/diagrams/06_phase2_harness.svg)

Numbers shown in the serialization-mode and surface-layer diagrams come from
the synthetic example report and are labelled as such.

## Layout

```
spec/        semantic_isa.md, surface_schema.md, grammar.ebnf,
             codec_contract.md, prereg.json
src/hsir/    lexicon (frozen rules) · state (shared transition rules) · parser
             serial (S0/S1/S1a/S2k/S3 modes) · surface (C sets) · realizer
             residual · codec (format v3, decodes v2)
             metrics (WB byte/symbol models, rank fits, bootstrap)
             dataset (loader, dedupe, split) · control (length-matched BPE)
             phase2 (report writer + decision logic) · synthetic (tests only)
experiments/ corpus_generator.py, entropy_report.py, rank_frequency.py
examples/    phase2_synthetic/ — full 11-file report from 2,000 synthetic docs
docs/        diagrams/ — Graphviz .dot sources + .png/.svg renders
.github/     workflows/tests.yml — tests on 3.10/3.12/3.13, freeze check, diagram render
```

## Arms

An arm is a serialization mode × surface set, preregistered in
`spec/prereg.json`:

* modes: `S0` ascii/explicit · `S3` compact/explicit · `S1` compact/implicit ·
  `S1a` ascii/implicit · `S2k4/8/16` implicit with state checkpoints
* surface sets: `C0` none · `C1` separators, serial comma · `C2` + articles,
  case · `C3` + verb variants · `C4` + reference forms · `C4P` permuted-id control

Arms `C0`–`C6` follow the v0.3.3 plan (`C0` = S1 with byte patches only;
`C5` = S2k8 + C4; `C6` = S1 + C4P). See `spec/semantic_isa.md` and
`spec/surface_schema.md`.

## Methodological guards built in

| Constraint | Where |
|---|---|
| Real data only for the decision; synthetic → `PENDING_REAL_DATA` | `phase2.py` |
| Whole-corpus accounting, fallback included; covered subset reported separately | `phase2.py` |
| Matched coding families (byte and model-facing symbol), most-severe decision wins | `metrics.py`, `phase2.py` |
| `L_shared` = rule tables + compiler source, amortized | `shared_artifact_costs.json` |
| No privileged fields (title, ingredients, NER) | `dataset.py`, tested |
| Exact + MinHash near-dup removal before a document-level split | `dataset.py` |
| Rules frozen and hashed; changes logged | `RULES_CHANGELOG.md`, `manifest.json` |
| Thresholds preregistered and hashed | `spec/prereg.json` |

## Freeze and Phase III prediction

v0.3.3 is frozen: `FREEZE.json` pins the compiler sources, rule tables,
surface schema, preregistration and `spec/phase3_prediction.md` (the
capacity-substitution hypothesis, written before any real-data result).
`experiments/freeze.py --verify` must pass before the real-data run; any
re-freeze is logged in `RULES_CHANGELOG.md`. Analysis clarifications made
after the freeze live in `spec/phase3_clarifications.md`, outside the frozen
set, and change no hypothesis, arm or threshold.

## Before running on real data

1. Freeze `spec/prereg.json` (thresholds, seed, orders). Its hash goes in the manifest.
2. Develop rule changes on the **train** split only; then score test once.
3. State-tracking accuracy is reported as `UNMEASURED` until a gold-annotated
   sample exists (~200 recipes would do).
