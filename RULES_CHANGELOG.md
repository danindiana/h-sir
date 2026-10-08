# Rule-table changelog

Every change to `src/hsir/lexicon.py` or to parser/realizer behaviour is logged
here with the version bump. Rules may only be developed against synthetic data
or the **train** split; once a real-data test split has been scored, the rules
used for that report are frozen (their SHA-256 is in `manifest.json`).

| Version | Change | Developed against |
|---|---|---|
| 0.3.0 | Initial recipe lexicon, versioned-entity grammar | synthetic |
| 0.3.1 | Verb-form words allowed as NP modifiers ("brown sugar", "cut beans") when the NP is headed or another NP word follows | synthetic |
| 0.3.2 | Lexical anaphor "the mixture" binds only to a composite; without one it is a new entity (was: fell back to last-touched entity, e.g. bound to "oven"). Pronouns keep the fallback | synthetic |
| surface schema 1 | Surface choice fields SEP, CASE, VERB, ART, LIST, REF with fixed value lists (`spec/surface_schema.md`); realizer defaults unchanged, so C0 output is byte-identical to 0.3.2 | synthetic |
| freeze 0.3.3 | Frozen before real data. Added order-0…6 sensitivity sweep (`sensitivity_orders`, `sensitivity_arms`) and `spec/phase3_prediction.md`; primary order 3 and all thresholds unchanged | synthetic |
| addendum (no re-freeze) | `spec/phase3_clarifications.md`: sign conventions for cost vs accuracy outcomes; model-size and context interactions reported separately; matched source documents and full-record source-byte likelihood in addition to matched FLOPs. `FREEZE.json` unchanged | review, before real data |
