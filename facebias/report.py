"""Plots and an HTML report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lr import ece, pav_llrs


def tippett(ax, llrs, y, label=None, color=None):
    llrs, y = np.asarray(llrs), np.asarray(y)
    for cls, ls in ((1, "-"), (0, "--")):
        v = np.sort(llrs[y == cls])
        ax.plot(v, 1 - np.arange(len(v)) / len(v), ls, color=color,
                label=f"{label} {'Hp' if cls else 'Hd'}" if label else ("Hp" if cls else "Hd"))
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("log10 LR"); ax.set_ylabel("proportion ≥ x")


def make_report(out_dir, pairs_llr, score_tab, score_sum, lr_tab, cross_fmr,
                group_col, system_info):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = pairs_llr.dropna(subset=["llr_global"])
    groups = sorted(d[f"{group_col}_a"].unique())
    cmap = plt.get_cmap("tab10")

    # 1 score distributions per group
    fig, axes = plt.subplots(len(groups), 1, figsize=(7, 1.8 * len(groups)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        s = d[d[f"{group_col}_a"] == g]
        ax.hist(s.score[s.mated == 0], bins=60, alpha=.6, density=True, label="non-mated")
        ax.hist(s.score[s.mated == 1], bins=60, alpha=.6, density=True, label="mated")
        ax.axvline(score_sum["threshold"], color="k", lw=.7)
        ax.set_ylabel(g, rotation=0, ha="right", fontsize=8)
    axes[0].legend(fontsize=8); axes[-1].set_xlabel("score")
    fig.tight_layout(); fig.savefig(out / "scores_per_group.png", dpi=130); plt.close(fig)

    # 2 Tippett per group (global system)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, g in enumerate(groups):
        s = d[d[f"{group_col}_a"] == g]
        tippett(ax, s.llr_global, s.mated, g, cmap(i % 10))
    ax.legend(fontsize=6, ncol=2); ax.set_title("Tippett per group - global LR system")
    fig.tight_layout(); fig.savefig(out / "tippett_per_group.png", dpi=130); plt.close(fig)

    # 3 ECE overall
    lo = np.linspace(-2.5, 2.5, 101)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lo, ece(d.llr_global, d.mated, lo), label="LR system")
    ax.plot(lo, ece(pav_llrs(d.llr_global, d.mated), d.mated, lo), "--", label="after PAV")
    ax.plot(lo, ece(np.zeros(len(d)), d.mated, lo), ":", label="LR=1")
    ax.set_xlabel("prior log10 odds"); ax.set_ylabel("ECE (bits)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "ece.png", dpi=130); plt.close(fig)

    # 4 calibration per group: mean llr in score bins vs PAV
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, g in enumerate(groups):
        s = d[d[f"{group_col}_a"] == g].sort_values("llr_global")
        ax.plot(s.llr_global, pav_llrs(s.llr_global, s.mated), color=cmap(i % 10), label=g, lw=1)
    lim = [d.llr_global.min(), d.llr_global.max()]
    ax.plot(lim, lim, "k:", lw=.8)
    ax.set_xlabel("reported log10 LR (global system)")
    ax.set_ylabel("PAV-calibrated log10 LR within group")
    ax.set_title("Calibration per group (on the diagonal = well calibrated)")
    ax.legend(fontsize=6); ax.set_ylim(lim[0] - 1, lim[1] + 1)
    fig.tight_layout(); fig.savefig(out / "calibration_per_group.png", dpi=130); plt.close(fig)

    # 5 cross-demographic FMR
    fig, ax = plt.subplots(figsize=(1 + .5 * len(cross_fmr.columns), 1 + .4 * len(cross_fmr)))
    im = ax.imshow(cross_fmr.values, cmap="magma")
    ax.set_xticks(range(len(cross_fmr.columns)), cross_fmr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cross_fmr.index)), cross_fmr.index, fontsize=6)
    ax.set_xlabel("probe group"); ax.set_ylabel("reference group")
    fig.colorbar(im, ax=ax, label="FMR"); fig.tight_layout()
    fig.savefig(out / "cross_demographic_fmr.png", dpi=130); plt.close(fig)

    score_tab.to_csv(out / "score_level_bias.csv")
    lr_tab.to_csv(out / "lr_level_bias.csv")
    cross_fmr.to_csv(out / "cross_demographic_fmr.csv")
    pairs_llr.to_csv(out / "pairs_with_llr.csv", index=False)
    (out / "summary.json").write_text(json.dumps({**score_sum, **system_info}, indent=2, default=float))

    fmt = lambda t: t.to_html(float_format=lambda x: f"{x:.4g}", border=0, classes="t")
    html = f"""<!doctype html><meta charset="utf-8"><title>Bias & LR report</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:2em auto;padding:0 1em}}
.t{{border-collapse:collapse;font-size:12px}}.t td,.t th{{padding:3px 6px;border-bottom:1px solid #ddd;text-align:right}}
img{{max-width:100%}}</style>
<h1>Demographic bias & LR validation</h1>
<p>System: <code>{json.dumps(system_info)}</code><br>
Grouping: <b>{group_col}</b>. Threshold at global FMR={score_sum['fmr_target']}: {score_sum['threshold']:.4g}.
FMR max/min = {score_sum['FMR_max/min']:.3g}, FNMR max/min = {score_sum['FNMR_max/min']:.3g},
FDR = {score_sum['FDR(alpha=0.5)']:.4f} (1 = fair).</p>
<h2>1. Score / decision level (same-demographic pairs)</h2>{fmt(score_tab)}
<img src="scores_per_group.png"><img src="cross_demographic_fmr.png">
<h2>2. LR level</h2>
<p><i>global_*</i>: one LR system calibrated on all groups (identity-disjoint CV).
<i>specific_*</i>: system calibrated per group. A clearly higher global_Cllr_cal than
specific_Cllr_cal for a group means a global calibration misstates LRs for that group.</p>
{fmt(lr_tab)}
<img src="calibration_per_group.png"><img src="tippett_per_group.png"><img src="ece.png">
"""
    (out / "report.html").write_text(html)
    return out / "report.html"
