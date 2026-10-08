# Recipe semantic ISA (rules v0.3.2, format v2)

The semantic stream `S` is canonical ASCII, one instruction per line,
single-space separated, `\n` terminated. It is produced by a deterministic,
rule-based parser (`src/hsir/parser.py`) from the `directions` field only.

## Instructions

| Opcode | Operands | Meaning |
|---|---|---|
| `S` | — | start of a supported sentence |
| `U` | — | unsupported sentence; realizer emits nothing, bytes go to `R` |
| `ENT` | `eN lexeme` | introduce entity identity `eN` (lexeme = words joined by `_`) |
| `QTY` | `ref num unit` | quantity on an argument; unit `-` = bare count |
| action | `ref*` | one of the action operators below |
| `DEST` | `prep ref` | destination / location |
| `TEMP` | `num unit` | temperature literal |
| `DUR` | `num unit` | duration literal |
| `OUT` | `ref` | resulting entity state |

Refs are versioned: `e3.2` = identity 3, state version 2.

Action operators (25): `PREHEAT CHOP SLICE GRATE PEEL MIX STIR WHISK ADD POUR
PLACE SPREAD SPRINKLE SEASON HEAT BOIL SIMMER COOK FRY BAKE COOL DRAIN COVER
SERVE WAIT`. Surface verbs map many-to-one (`dice`, `mince` → `CHOP`).

## State semantics

| Class | Operators | Effect |
|---|---|---|
| combining (≥2 args) | `MIX WHISK` | new composite entity `mixture`, version 0 |
| absorbing (with `DEST`) | `ADD POUR PLACE SPREAD SPRINKLE` | destination version +1; becomes focus |
| transforming | `CHOP … WAIT` | each argument version +1 |
| observing | `PREHEAT SERVE` | no state change |

Anaphora (`it`, `them`, `mixture`, `everything`, `all`) bind to the current
composite focus, else the most recently touched entity. This is a recorded
interpretation, not a claim of correctness; accuracy needs gold annotation.

A sentence that fails any rule becomes `U` and leaves entity state untouched.
Zero-argument actions (`Bake for 30 minutes.`) do **not** infer an implicit
object.

## Serialization modes (`src/hsir/serial.py`)

The instruction list above is the canonical form. It can be written to the
stream in several modes; every mode decodes back to the identical canonical
list, validated by replaying `state.transition`, and must re-encode to the
exact same bytes (canonical-form check).

| Mode | Encoding | State | What is serialized |
|---|---|---|---|
| S0 | ascii | explicit | everything above (v0.3.1 format, byte-identical) |
| S3 | compact | explicit | same information; opcode bytes, varints, NUL-terminated strings |
| S1 | compact | implicit | `ENT lexeme`, entity ids on refs; no versions, no composite `ENT`, no `OUT` |
| S1a | ascii | implicit | S1 information, S0-style spelling (diagnostic) |
| S2k*k* | compact | checkpoint | S1 + `CK` snapshot of every entity's version after every *k*-th action |

Derived content (versions, composite `ENT`, `OUT`) is a deterministic function
of the preceding stream, so it carries zero information in principle. Its
measured coding cost under a finite-context model is therefore a
*learnability* cost: how hard it is for that predictor to track state.

A `CK` that disagrees with the replayed state raises `checkpoint mismatch`;
a missing or extra `CK` fails the canonical-form check.

Contrasts (preregistered in `spec/prereg.json`):

* S0 − S3: serialization overhead of explicit state
* S3 − S1: cost of explicit state, compact spelling
* S1a − S1 and S0 − S1a: the same split taken along the other path
* S2k − S1: checkpoint overhead
