# Diagrams

Graphviz sources (`.dot`) with rendered `.png` (160 dpi) and `.svg`. Edit the
`.dot`, then re-render everything with:

```bash
sudo apt-get install -y graphviz   # once
bash docs/diagrams/render.sh
```

or a single file: `dot -Tpng -Gdpi=160 01_codec_pipeline.dot -o 01_codec_pipeline.png`.
Prefer the SVGs for slides and editing.

| # | Diagram | Shows |
|---|---|---|
| 01 | [codec pipeline](01_codec_pipeline.svg) | encoder and decoder paths, Mode A fallback, where S, C and R come from |
| 02 | [container format v3](02_container_format_v3.svg) | byte layout, append-only mode/set ids, v2 compatibility, residual wire ops |
| 03 | [entity state machine](03_entity_state_machine.svg) | versioned refs, the four transition classes, reference resolution, who calls `state.transition` |
| 04 | [serialization modes](04_serialization_modes.svg) | spelling × state grid (S0, S3, S1, S1a, S2k*) with measured contrasts |
| 05 | [surface layer](05_surface_layer.svg) | slot schema, sparse encoding, C0 → C4 cost/savings, permuted-id control |
| 06 | [Phase II harness](06_phase2_harness.svg) | inputs, dedupe/split, 13 arms, matched coders, bootstrap, report files |
| 07 | [decision logic](07_decision_logic.svg) | preregistered GO / REVISE / STOP / INVALID / PENDING rules |
| 08 | [research roadmap](08_research_roadmap.svg) | what is done, the pending real-data gate, and planned (not run) Phase III |

**Numbers in 04 and 05 come from the synthetic example report**
(`examples/phase2_synthetic/`, symbol coder, order 3). They are labelled as
synthetic in the diagrams themselves and are not real-data evidence. Diagram 08
marks Phase III as planned and not run.
