# Phase III analysis clarifications (addendum, not a protocol change)

Recorded 2026-10-08, after the v0.3.3 freeze and **before any real-data
result**. These clarify how the frozen prediction is analysed and reported.
They add no new hypotheses, arms, thresholds or outcomes.

Applies to `spec/phase3_prediction.md`, frozen at SHA-256
`4010ddb33e5f9c3eab9578f8e7f7c80df49c12f778afb06da83f5daa0b656d29`.
This file is deliberately outside the frozen set; `FREEZE.json` is unchanged.

## A1. Sign conventions

* **γ-type outcome** (hybrid cost relative to BPE; larger = worse for
  H-SIR): capacity substitution predicts a **positive** interaction with
  log(params) and with log(context). This is what the frozen text states.
* **Accuracy-type outcomes**, with advantage A = metric(H-SIR) − metric(BPE)
  where larger metric = better: the same hypothesis predicts A **decreases**,
  i.e. a **negative** interaction. Every reported coefficient states which of
  the two conventions it uses.

## A2. Size and context are separate mechanisms

The representation × log(params) and representation × log(context) terms are
estimated and reported separately, each with its own 95% interval. A term
supports H-P1 for its own mechanism only. Support from one term is reported as
"supported for model size" or "supported for context length", never as support
for capacity substitution in general. Both terms must meet the frozen criterion
for an unqualified "supported".

## A3. Matched FLOPs is necessary but not sufficient

Every arm in a cell:

* trains on the **same set of original documents**, each seen the same number
  of times (a document-exposure schedule, not a token budget), and reports
  both training tokens and source bytes actually seen;
* is evaluated on the **same held-out original documents**, scored as
  source-byte-normalized likelihood of the full record (S + C + R for hybrids,
  so nothing is dropped from the hybrid's cost);
* treats shared information identically: `L_shared` is either excluded from
  every arm or amortized with the same rule for every arm.

If the FLOPs budget cannot cover the same source documents for a longer
representation, both a matched-FLOPs run and a matched-source-documents run
are reported, and the primary test uses matched source documents.
