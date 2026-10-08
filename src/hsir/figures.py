"""Static report figures (optional; skipped if matplotlib is unavailable).

order_sweep.png: gamma (hybrid / baseline) against context-model order, one
panel per coding family, one line per sensitivity arm, 95% paired-bootstrap
intervals as thin bars, parity (gamma = 1) as a reference line. Lines are
direct-labelled and carry distinct markers, so identity is never colour alone.
"""
from __future__ import annotations

from pathlib import Path

# Reference categorical palette, slots 1-4 in fixed order (validated: adjacent
# CVD dE >= 9.1, normal-vision dE >= 22.9 on #fcfcfb; slots 3-4 sit below 3:1
# contrast, so every line is direct-labelled and the data ships as JSON).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def order_sweep(sweep: dict, out: Path, primary_order: int) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None
    fams = list(sweep["by_family"])
    fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 4.2),
                             sharey=True, facecolor=SURFACE)
    if len(fams) == 1:
        axes = [axes]
    for ax, fam in zip(axes, fams):
        data = sweep["by_family"][fam]
        orders = sweep["orders"]
        ax.set_facecolor(SURFACE)
        ax.axhline(1.0, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(orders[-1], 1.0, "parity (γ = 1) ", va="top", ha="right",
                fontsize=8, color=INK_2)
        ax.axvline(primary_order, color=GRID, lw=6, zorder=0)
        ends = []
        for i, arm in enumerate(sweep["arms"]):
            g = [data[f"order_{k}"][arm]["gamma"] for k in orders]
            lo = [data[f"order_{k}"][arm]["gamma_ci95"][0] for k in orders]
            hi = [data[f"order_{k}"][arm]["gamma_ci95"][1] for k in orders]
            col = SERIES[i % len(SERIES)]
            ax.plot(orders, g, color=col, lw=2, marker=MARKERS[i % 4], ms=7,
                    mec=SURFACE, mew=1.5, zorder=3, label=arm)
            ax.vlines(orders, lo, hi, color=col, lw=1, zorder=2)
            ends.append([g[-1], arm])
        # direct end labels, dodged so they never overlap
        lo_y, hi_y = ax.get_ylim()
        gap = 0.055 * (hi_y - lo_y)
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y, arm in ends:
            ax.text(orders[-1] + 0.15, y, arm, va="center", ha="left",
                    fontsize=9, color=INK)
        ax.set_title(f"{fam} coder", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("context-model order k", fontsize=9, color=INK_2)
        ax.set_xticks(orders)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xlim(orders[0] - 0.3, orders[-1] + 0.9)
    axes[0].set_ylabel("γ = hybrid bits / baseline bits (lower is better)",
                       fontsize=9, color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Hybrid cost vs coder context (shaded column: preregistered "
                 f"primary order {primary_order})", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    path = out / "order_sweep.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path.name
