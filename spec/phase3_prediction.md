# Phase III preregistered prediction (written before any real-data result)

Status: **registered 2026-10-08, protocol v0.3.3, before the RecipeNLG run.**
Nothing here authorizes training; the Phase II gate still decides that.

## Motivation (not evidence)

In the synthetic Phase II order sweep, the hybrid representation beats the
byte baseline only for weak context models (symbol coder, γ < 1 at orders
0–2) and loses as order grows (Spearman ρ(γ, order) ≈ 0.89). The surface
layer's own gain (C4 − C0) stays roughly constant across orders. An order-k
n-gram is not a transformer, so this motivates a prediction; it does not
support one.

## Hypothesis H-P1 (capacity substitution)

Explicit semantic structure substitutes for predictive context or model
capacity. Therefore the H-SIR advantage over BPE, at matched training FLOPs,
**shrinks as model size or context length grows**.

## Design

| Factor | Levels |
|---|---|
| Representation | BPE baseline · length-matched nonsemantic BPE control · C0 · C4 |
| Model size | ~2M · ~8M · ~30M parameters (decoder-only) |
| Context length | 128 · 512 tokens |
| Seeds | 3 per cell |

Same documents, split and FLOPs budget per (size, context) cell.

## Outcomes

* **Primary:** held-out source bits per byte of the full record
  (S + C + R for hybrids), i.e. Phase II's γ measured by a trained model.
* **Secondary:** entity-state probe accuracy (needs ~200 gold-annotated
  recipes); held-out lexical-combination continuation.

## Test and falsification

Fit `metric ~ representation × log(params) + representation × log(context)`
across cells. H-P1 predicts a **positive** interaction for C4 vs BPE on the
γ-type outcome (the hybrid's relative cost rises with capacity).

* Supported: interaction > 0 with a 95% CI excluding 0, and C4 better than
  the nonsemantic control at the smallest cell.
* Falsified: interaction CI includes 0 or is negative, or C4 never beats the
  nonsemantic control in any cell.

## Entry conditions

* Compression track: real-data Phase II decision `GO` for the arm used.
* Inductive-bias track (if compression is `REVISE`/`STOP`): a written pivot
  decision citing the real-data report, recorded before training starts.
