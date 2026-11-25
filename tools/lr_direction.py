"""
Direction of LR mis-calibration per group: are a globally calibrated system's LRs too
high or too low for a group, compared with a group-specific calibration?

For every condition class and group (same-demographic pairs, identity-disjoint CV):
  bias_mated    = mean(log10 LR_global - log10 LR_group) over mated pairs
  bias_nonmated = same over non-mated pairs
  > 0  : the global system overstates the evidence for that group (LRs too high:
         towards 'same source'), < 0 : understates it.
  plus the rates of misleading evidence under both systems:
  RMED (non-mated with LR > 1) and RMEP (mated with LR < 1), and non-mated LR > 100.

    python tools/lr_direction.py                             # adaface (with rescued faces if available)
    python tools/lr_direction.py --scored data/scored_system_c.csv --name system_c
Output: results_lr_direction_<name>.md / .csv / .png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from facebias.lr import cross_validated_llrs  # noqa: E402
from analyze_by_condition import CLASSES  # noqa: E402

GROUPS = ["ancestry", "gender_presentation", "age_band"]


def direction(pairs, name, transform):
    pairs = pairs.dropna(subset=["score"]).copy()
    pairs["same_demo"] = pairs.same_demo.astype(str).str.lower().eq("true")
    rows = []
    for cls, sel in CLASSES.items():
        d = pairs[sel(pairs.condition_b.astype(str)) & pairs.same_demo].reset_index(drop=True)
        if d.mated.nunique() < 2:
            continue
        d["llr_g"] = cross_validated_llrs(d.score, d.mated, d.id_a, d.id_b,
                                          calibrator="logit", transform=transform)
        for g in GROUPS:
            for grp, dg in d.groupby(f"{g}_a"):
                if dg.mated.nunique() < 2 or dg.mated.sum() < 20:
                    continue
                spec = cross_validated_llrs(dg.score, dg.mated, dg.id_a, dg.id_b,
                                            calibrator="logit", transform=transform)
                dg = dg.assign(llr_s=spec).dropna(subset=["llr_g", "llr_s"])
                m, n = dg[dg.mated == 1], dg[dg.mated == 0]
                rows.append({
                    "model": name, "class": cls, "group_by": g, "group": grp,
                    "n_mated": len(m), "n_nonmated": len(n),
                    "bias_mated": (m.llr_g - m.llr_s).mean(),
                    "bias_nonmated": (n.llr_g - n.llr_s).mean(),
                    "RMED_global": (n.llr_g > 0).mean(), "RMED_group": (n.llr_s > 0).mean(),
                    "RMEP_global": (m.llr_g < 0).mean(), "RMEP_group": (m.llr_s < 0).mean(),
                    "nonmated_LR>100_global": (n.llr_g > 2).mean(),
                    "nonmated_LR>100_group": (n.llr_s > 2).mean(),
                })
        print(f"{name}: {cls} done", flush=True)
    return pd.DataFrame(rows)


def center(res):
    """Group-specific models are fitted on fewer pairs, so with (near-)separable classes they
    give less extreme LRs than the global model for *every* group (regularisation/ELUB).
    That common offset is not group bias; the relative bias (minus the mean over the groups
    of the same class and grouping) is what differs between groups."""
    for c in ["bias_mated", "bias_nonmated"]:
        res[c + "_rel"] = res[c] - res.groupby(["class", "group_by"])[c].transform("mean")
    return res


def report(res, name):
    res = center(res)
    res.to_csv(ROOT / f"results_lr_direction_{name}.csv", index=False)
    md = [f"# Richting van LR-miskalibratie: {name}\n",
          "bias = gemiddelde (log10 LR globaal − log10 LR groepsspecifiek). **> 0: het globale systeem geeft voor "
          "die groep te hoge LR's** (bewijs te sterk richting 'zelfde bron'); < 0: te laag. RMED = aandeel "
          "non-mated paren met LR > 1, RMEP = aandeel mated paren met LR < 1 (misleidend bewijs), "
          "globaal vs groepsspecifiek gekalibreerd. Identiteit-disjuncte kruisvalidatie, alleen paren binnen "
          "dezelfde demografische groep.\n",
          "**_rel** = bias min het gemiddelde over de groepen binnen dezelfde conditie: groepsspecifieke modellen "
          "worden op minder paren gefit en geven daardoor bij goed scheidbare condities voor *alle* groepen minder "
          "extreme LR's. Die gemeenschappelijke verschuiving is geen groepsbias; de _rel-kolommen en de figuur "
          "tonen het verschil tussen groepen.\n"]
    for cls in res["class"].unique():
        md.append(f"\n## {cls}\n")
        for g in GROUPS:
            t = res[(res["class"] == cls) & (res.group_by == g)].set_index("group")
            if t.empty:
                continue
            t = t[["n_mated", "bias_mated_rel", "bias_nonmated_rel", "bias_mated", "bias_nonmated", "RMED_global", "RMED_group",
                   "RMEP_global", "RMEP_group", "nonmated_LR>100_global", "nonmated_LR>100_group"]]
            t = t.sort_values("bias_mated_rel", ascending=False)
            md.append(f"\n### {g}\n")
            try:
                md.append(t.round(3).to_markdown())
            except ImportError:
                md.append("```\n" + t.round(3).to_string() + "\n```")
    (ROOT / f"results_lr_direction_{name}.md").write_text("\n".join(md), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = list(res["class"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 7), sharey=True)
    axes = np.atleast_1d(axes)
    anc = res[res.group_by == "ancestry"]
    order = (anc.groupby("group").bias_mated_rel.mean().sort_values().index.tolist())
    y = np.arange(len(order))
    for ax, cls in zip(axes, classes):
        t = anc[anc["class"] == cls].set_index("group").reindex(order)
        ax.barh(y - 0.2, t.bias_nonmated_rel, height=0.4, label="non-mated", color="#c0504d")
        ax.barh(y + 0.2, t.bias_mated_rel, height=0.4, label="mated", color="#4f81bd")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(cls); ax.set_xlabel("relatieve bias log10 LR")
    axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=8)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{name}: log10 LR globaal − groepsspecifiek, t.o.v. gemiddelde groep  (> 0 = globale LR relatief te hoog)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / f"results_lr_direction_{name}.png", dpi=130)
    print(f"-> results_lr_direction_{name}.md/.csv/.png", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scored")
    p.add_argument("--name")
    p.add_argument("--transform", default=None, help="none for cosine (default), logit for System C [0,1]")
    a = p.parse_args(argv)
    if a.scored:
        src, name = Path(a.scored), a.name or Path(a.scored).stem
        transform = a.transform or "logit"
    else:
        src = ROOT / "data/scored_adaface_rescued.csv"
        if not src.exists():
            src = ROOT / "data/scored_adaface.csv"
        name, transform = a.name or src.stem.replace("scored_", ""), a.transform or "none"
    res = direction(pd.read_csv(src), name, transform)
    report(res, name)


if __name__ == "__main__":
    main()
