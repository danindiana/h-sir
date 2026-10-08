# Surface choice schema (C), schema version 1

`ŷ = G(S, C)`, `R = diff(ŷ, x)`, `D(S, C, R) = x`.

C is sparse: slots are enumerated deterministically from the canonical
instruction list and the enabled field set; a choice is stored only when its
value differs from the default (index 0). Wire format per stored choice:

```
varint gap      # slots skipped since the previous stored choice
varint value    # value index (permuted in C4P)
```

The decoder rejects: a stored default, an out-of-range value, a gap past the
last slot, or any C in C0. R still guarantees exactness, so a missing or
unrepresentable choice costs bits, never correctness.

## Slots (in enumeration order, per supported sentence j)

| Field | Slot | Values (index 0 = default) |
|---|---|---|
| `SEP` | before sentence j > 0 | `" "`, `"\n"`, `""`, `"\n\n"`, `"  "`, `" \n"`, `"\n "`, `"\r\n"` |
| `CASE` | verb initial | upper, lower |
| `VERB` | verb surface (ops with >1 form) | the op's surface forms, in lexicon order |
| `REF` | each argument, then destination | NP, it, them, everything, all |
| `ART` | each argument/destination without quantity | the, a, an, none, some, your |
| `LIST` | argument lists of ≥3 items | Oxford comma, plain |

Within a sentence the order is SEP, CASE, VERB, then per argument REF, ART,
then LIST, then destination REF, ART. Value lists are append-only; changing
an existing entry requires a schema version bump.

## Observation

The parser records what the source actually used (verb word, determiner,
pronoun, list punctuation, sentence-initial case, inter-sentence whitespace).
A value outside the schema counts as **unrepresentable** (fallback to byte
patches). A slot with nothing to observe — a leading adverb hides the verb's
case, or an unsupported sentence intervenes between two separators — counts as
**unobserved**. Both rates are reported per field.

## Surface sets

| Set | Fields |
|---|---|
| C0 | none (v0.3.2 behaviour) |
| C1 | SEP, LIST |
| C2 | C1 + ART, CASE |
| C3 | C2 + VERB |
| C4 | C3 + REF |
| C4P | C4 with value ids reversed per slot (`v' = n − 1 − v`), a fixed bijection |
