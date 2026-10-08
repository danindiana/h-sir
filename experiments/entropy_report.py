"""Print a compact summary of a Phase II report directory.

    python experiments/entropy_report.py phase2/
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
j = lambda n: json.loads((d / n).read_text())  # noqa: E731
feas, cov, cod, boot = (j("feasibility_decision.json"), j("parser_coverage.json"),
                        j("coding_rates.json"), j("bootstrap_intervals.json"))
prim = f"order_{j('manifest.json')['coding_models']['primary_order']}"
print(f"decision: {feas['decision']}  "
      f"(diagnostic: {feas['diagnostic_decision_ignoring_synthetic_flag']}; "
      f"eligible: {feas['eligible_arms_for_phase_III'] or 'none'})")
t = cov["test"]
print(f"coverage  doc(B)={t['doc_mode_B_fraction']:.3f}  "
      f"doc(full)={t['doc_fully_covered_fraction']:.3f}  "
      f"sent={t['sentence_coverage']:.3f}  byte={t['byte_weighted_coverage']:.3f}")
for fam, by_arm in cod["results"].items():
    print(f"\n{fam} family, {prim} (bits per source byte)")
    print(f"  {'arm':6s} {'base':>6s} {'S':>6s} {'C|S':>6s} {'R|S,C':>6s} {'gamma':>6s}"
          f"  95% CI          rho_R  decision")
    for a, orders in by_arm.items():
        v = orders[prim]
        ci = boot[fam][f"{a}.gamma_no_shared"]["ci95"]
        dec = feas["by_arm"][a]["by_family"][fam]["decision"]
        print(f"  {a:6s} {v['baseline_bits_per_source_byte']:6.3f} "
              f"{v['S_bits_per_source_byte']:6.3f} {v['C_bits_per_source_byte']:6.3f} "
              f"{v['R_bits_per_source_byte']:6.3f} {v['gamma_no_shared']:6.3f}  "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]  {v['rho_R']:.3f}  {dec}")
    for row in cod["arm_decomposition"]["by_family"][fam]:
        key = f"delta_gamma[{row['contrast'].replace(' ', '')}]"
        ci = boot[fam][key]["ci95"]
        print(f"  {row['contrast']:12s} dS {row['delta_S_bpb']:+.3f} "
              f"dC {row['delta_C_bpb']:+.3f} dR {row['delta_R_bpb']:+.3f} "
              f"dgamma {row['delta_gamma']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
              f"{row['meaning']}")
usage = j("residual_allocation.json")["surface_choice_usage_test_split"]
if usage:
    cs = max(usage, key=lambda k: len(usage[k]["fields"]))
    print(f"\nsurface choices ({cs}, test split)")
    for f, v in usage[cs]["fields"].items():
        print(f"  {f:5s} slots {v['slots']:6d}  emitted {v['emit_rate']:.3f}  "
              f"unobserved {v['unobserved_rate']:.3f}  "
              f"fallback {v['fallback_rate_unrepresentable']:.3f}")

sweep = cod.get("order_sweep", {})
if sweep.get("by_family"):
    print(f"\norder sweep (gamma; primary order {sweep['primary_order']}; "
          f"sensitivity only)")
    for fam, rows in sweep["by_family"].items():
        print(f"  {fam}: " + "  ".join(f"k={k}" for k in sweep["orders"]))
        for a in sweep["arms"]:
            print(f"    {a:5s} " + "  ".join(
                f"{rows[f'order_{k}'][a]['gamma']:.2f}" for k in sweep["orders"])
                + f"   beats baseline at k = "
                f"{rows['summary'][a]['orders_where_hybrid_beats_baseline_ci'] or 'none'}")
