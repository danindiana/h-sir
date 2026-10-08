"""Print the Zipf / stretched-exponential comparison from a Phase II report.

    python experiments/rank_frequency.py phase2/
"""
import json
import sys
from pathlib import Path

rf = json.loads((Path(sys.argv[1]) / "rank_frequency.json").read_text())


def show(name, r):
    if r.get("status") != "ok":
        print(f"{name:28s} {r.get('status')}")
        return
    print(f"{name:28s} types={r['n_types_train']:5d}  "
          f"zipf a={r['zipf']['alpha']:.2f} {r['zipf']['test_bits_per_token']:.3f}b  "
          f"sexp {r['stretched_exp']['test_bits_per_token']:.3f}b  "
          f"emp {r['empirical_add_half']['test_bits_per_token']:.3f}b  "
          f"-> {r['preferred_parametric_on_test']}"
          + ("  (caution: few types)" if "caution" in r else ""))


show("H_Z1 operators", rf["H_Z1_operators"])
for c, r in rf["H_Z2_operands"].items():
    show(f"H_Z2 operands/{c}", r)
for m, r in rf["H_Z3_serialized_symbols"].items():
    show(f"H_Z3 symbols/{m}", r)
ce = rf["conditional_entropy_concept_given_action"]
if ce.get("status") == "ok":
    h = ce["heldout_cross_entropy_bits"]
    print(f"H(concept)={h['H(Y)']:.3f}b  H(concept|action)={h['H(Y|C)']:.3f}b  "
          f"gain={h['gain']:.3f}b (held-out)")
