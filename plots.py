"""Static figures for the README/report (matplotlib, light theme, validated categorical palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def forest_relative_lift(rows, path, title):
    """rows: list of dicts with arm, metric, rel_lift, rel_ci_low, rel_ci_high."""
    metrics = ["visit", "conversion", "spend"]
    arms = ["Mens E-Mail", "Womens E-Mail"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for j, arm in enumerate(arms):
        for i, m in enumerate(metrics):
            r = next(r for r in rows if r["arm"] == arm and r["metric"] == m)
            y = i + (0.18 if j == 0 else -0.18)
            ax.plot([100 * r["rel_ci_low"], 100 * r["rel_ci_high"]], [y, y], color=SERIES[j], lw=2,
                    solid_capstyle="round")
            ax.plot(100 * r["rel_lift"], y, "o", color=SERIES[j], ms=8, mec=SURFACE, mew=2,
                    label=arm if i == 0 else None)
            ax.text(100 * r["rel_ci_high"] + 3, y, f"{100 * r['rel_lift']:+.0f}%", va="center",
                    color=INK2, fontsize=9)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(metrics)), [m.capitalize() for m in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("Relative lift vs. No E-Mail (%, 95% CI)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def variance_reduction_plot(rows, path, title):
    """rows: list of dicts with label, method, variance_reduction_se (1 - (SE/SE_unadjusted)^2)."""
    labels = list(dict.fromkeys(r["label"] for r in rows))
    methods = ["cuped", "regression_adjusted", "cupac"]
    names = {"cuped": "CUPED (past-year spend)", "regression_adjusted": "Regression adj. (all pre-period covariates)",
             "cupac": "CUPAC (cross-fitted GBM prediction)"}
    fig, ax = plt.subplots(figsize=(7.2, 4))
    h = 0.26
    y = np.arange(len(labels))
    for k, m in enumerate(methods):
        vals = [100 * next(r["variance_reduction_se"] for r in rows if r["label"] == lab and r["method"] == m)
                for lab in labels]
        ax.barh(y + (k - 1) * h, vals, height=h - 0.04, color=SERIES[k], label=names[m])
        for yi, v in zip(y + (k - 1) * h, vals):
            ax.text(v + 0.04, yi, f"{v:.2f}%" if v < 0.1 else f"{v:.1f}%", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(3.5, ax.get_xlim()[1]))
    ax.set_xlabel("Variance reduction vs. unadjusted estimate (%)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def qini_plot(curves, path, title):
    """curves: dict name -> (phi, q) with q = incremental visits per customer."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, (name, (phi, q)) in enumerate(curves.items()):
        ax.plot(100 * phi, 1000 * q, color=SERIES[k], lw=2, label=name)
    phi, q = next(iter(curves.values()))
    ax.plot([0, 100], [0, 1000 * q[-1]], color=MUTED, lw=1.2, ls="--", label="Random targeting")
    ax.set_xlabel("Share of customers e-mailed, ranked by predicted uplift (%)")
    ax.set_ylabel("Incremental visits per 1,000 customers")
    ax.set_title(title)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def decile_plot(rows, path, title, model_name):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    d = np.array([r["decile"] for r in rows])
    obs = np.array([100 * r["obs_uplift"] for r in rows])
    se = np.array([100 * r["se"] for r in rows])
    pred = np.array([100 * r["pred_uplift"] for r in rows])
    ax.bar(d, obs, width=0.7, color=SERIES[0], label="Observed uplift (held-out, ±1.96 SE)")
    ax.errorbar(d, obs, yerr=1.96 * se, fmt="none", ecolor=INK2, elinewidth=1, capsize=0)
    ax.plot(d, pred, "o-", color=SERIES[1], lw=2, ms=6, mec=SURFACE, mew=1.5, label=f"Predicted uplift ({model_name})")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(d)
    ax.set_xlabel("Predicted-uplift decile (1 = highest)")
    ax.set_ylabel("Visit-rate uplift (pp)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
