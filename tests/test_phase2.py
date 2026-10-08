"""End-to-end Phase II contract on synthetic and on a tiny RecipeNLG-shaped CSV."""
import csv
import json

from hsir import phase2

FILES = ["manifest.json", "dataset_audit.json", "parser_coverage.json",
         "roundtrip_results.json", "coding_rates.json",
         "residual_allocation.json", "rank_frequency.json",
         "bootstrap_intervals.json", "shared_artifact_costs.json",
         "failure_cases.jsonl", "feasibility_decision.json"]


def test_synthetic_report_contract(tmp_path):
    phase2.main(["--synthetic", "300", "--out", str(tmp_path)])
    for f in FILES:
        assert (tmp_path / f).exists(), f
    feas = json.loads((tmp_path / "feasibility_decision.json").read_text())
    assert feas["decision"] == "PENDING_REAL_DATA"
    assert feas["model_training_permitted"] is False
    rt = json.loads((tmp_path / "roundtrip_results.json").read_text())
    assert rt["all_exact"]
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["synthetic"] is True and len(man["prereg_sha256"]) == 64
    sweep = json.loads((tmp_path / "coding_rates.json").read_text())["order_sweep"]
    assert sweep["orders"] == [0, 1, 2, 3, 4, 5, 6]
    assert set(sweep["by_family"]) == {"byte", "symbol"}
    row = sweep["by_family"]["symbol"]["order_3"]["C4"]
    lo, hi = row["gamma_ci95"]
    assert lo <= row["gamma"] <= hi


def test_csv_loader_uses_only_directions(tmp_path):
    path = tmp_path / "mini.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "title", "ingredients", "directions", "link",
                    "source", "NER"])
        for i in range(40):
            w.writerow([i, f"SECRET TITLE {i}", '["1 c. SECRET"]',
                        json.dumps([f"Mix flour and sugar {i}.",
                                    "Bake for 20 minutes."]),
                        f"example.com/{i}", "Gathered", '["secret"]'])
    out = tmp_path / "out"
    phase2.main(["--input", str(path), "--out", str(out)])
    feas = json.loads((out / "feasibility_decision.json").read_text())
    assert feas["decision"] in {"GO", "REVISE", "STOP"}
    blob = "".join((out / f).read_text() for f in FILES)
    assert "SECRET" not in blob
    audit = json.loads((out / "dataset_audit.json").read_text())
    assert audit["fields_used"] == ["directions"]


def test_freeze_manifest_intact():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "experiments" / "freeze.py"),
                        "--verify"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
