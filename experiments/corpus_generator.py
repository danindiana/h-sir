"""Write a synthetic JSONL corpus. FOR UNIT TESTS / DEBUGGING ONLY.

    python experiments/corpus_generator.py --n 1000 --out synth.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsir import synthetic  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
with open(a.out, "w") as f:
    for i, t in enumerate(synthetic.corpus(a.n, a.seed)):
        f.write(json.dumps({"link": f"synthetic-{i}", "synthetic": True,
                            "directions": t.split("\n")}) + "\n")
print(f"wrote {a.n} synthetic docs to {a.out} (not valid for feasibility)")
