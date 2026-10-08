"""Write or verify the v0.3.3 freeze manifest (FREEZE.json).

    python experiments/freeze.py --verify     # exit 1 if anything changed
    python experiments/freeze.py --write      # (re)freeze; log why in RULES_CHANGELOG.md

Frozen: compiler sources, rule tables, surface schema, preregistration, and
the Phase III prediction. Run --verify before the real-data Phase II run.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsir import codec, lexicon, phase2, surface  # noqa: E402

FILES = ([f"src/hsir/{f}" for f in phase2.COMPILER_SOURCES]
         + ["spec/prereg.json", "spec/surface_schema.md",
            "spec/phase3_prediction.md"])


def current() -> dict:
    return {"protocol_version": "0.3.3",
            "format_version": codec.FORMAT_VERSION,
            "rules_version": lexicon.RULES_VERSION,
            "surface_schema_version": surface.SURFACE_SCHEMA_VERSION,
            "rule_table_sha256": hashlib.sha256(lexicon.table_bytes()).hexdigest(),
            "files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                      for f in FILES}}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    path = ROOT / "FREEZE.json"
    cur = current()
    if a.write:
        path.write_text(json.dumps(cur, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path.name}")
        return 0
    frozen = json.loads(path.read_text())
    bad = [k for k in cur if k != "files" and cur[k] != frozen.get(k)]
    bad += [f for f in FILES if cur["files"][f] != frozen["files"].get(f)]
    if bad:
        print("FREEZE VIOLATED:", ", ".join(bad))
        return 1
    print(f"freeze intact: {len(FILES)} files, rules {cur['rules_version']}, "
          f"surface schema {cur['surface_schema_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
