"""
Precise version of the bias and LR-direction results, with uncertainty, for every matcher.

For each matcher (data/scored_<m>_v2.csv), condition class and group:
  - FMR / FNMR at the class threshold (FMR = 1e-3 over all pairs of the class) with a 95% CI
    from an identity-clustered bootstrap (resampling reference persons, 2000 replicates);
  - FMR ratio of the group vs the class as a whole, with CI;
  - FMR / FNMR on balanced subsamples (equal number of persons per group), 50 draws;
  - LR direction: out-of-fold log10 LRs of one global calibration vs a group-specific calibration
    (identity-disjoint CV, 10 repeats), mean difference for mated and non-mated pairs, relative
    to the mean over the groups, with joint bootstrap CIs (2000 replicates).

    python tools/precise_stats.py [--models adaface ...] [--boot 2000] [--repeats 10]
Output: results_precise/<m>_fmr.csv, <m>_lr.csv, summary.md, forest_<class>.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "results_precise"
GROUPS = ["ancestry", "gender_presentation", "age_band"]
CLASSES = ["passfoto", "edit_hq", "cctv_still", "video_I", "video_P"]


def log(m):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True)


def md_table(df, d=4):
    try:
        return df.round(d).to_markdown()
    except ImportError:
        return "```\n" + df.round(d).to_string() + "\n```"


def per_person(d, col, value):
    """Per reference person: (sum of value, count) for rows selected by col (bool mask)."""
    g = d[col].groupby(d.id_a)
    return g.sum(), g.size()


def boot_ratio(num, den, B, rng):
    """Identity-clustered bootstrap of sum(num)/sum(den) (arrays per person)."""
    n = len(num)
    if n == 0:
        return np.full(B, np.nan)
    w = rng.multinomial(n, np.full(n, 1 / n), size=B)          # B x n resampling weights
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ num) / (w @ den)


def fmr_part(p, name, B, rng):
    from analyze_by_condition import CLASSES as SEL
    from facebias.bias import threshold_at_fmr
    rows = []
    p = p.dropna(subset=["score"])
    for cls in CLASSES:
        d = p[SEL[cls](p.condition_b.astype(str))]
        if d.mated.nunique() < 2:
            continue
        thr = threshold_at_fmr(d.score, d.mated, 1e-3)
        same = d[d.sd]
        nm, m = same[same.mated == 0], same[same.mated == 1]
        cls_fmr = (nm.score >= thr).mean()
        for g in GROUPS:
            col = f"{g}_a"
            # balanced subsamples: equal number of reference persons per group
            ids = same.groupby(col).id_a.unique()
            nmin = min(len(v) for v in ids)
            for grp in ids.index:
                a = nm[nm[col] == grp]; b = m[m[col] == grp]
                fa = (a.score >= thr).groupby(a.id_a).agg(["sum", "size"])
                fb = (b.score < thr).groupby(b.id_a).agg(["sum", "size"])
                bf = boot_ratio(fa["sum"].values.astype(float), fa["size"].values.astype(float), B, rng)
                bn = boot_ratio(fb["sum"].values.astype(float), fb["size"].values.astype(float), B, rng)
                bal_f, bal_n = [], []
                for s in range(50):
                    pick = set(np.random.default_rng(s).choice(ids[grp], nmin, replace=False))
                    aa, bb = a[a.id_a.isin(pick)], b[b.id_a.isin(pick)]
                    bal_f.append((aa.score >= thr).mean()); bal_n.append((bb.score < thr).mean())
                fmr = fa["sum"].sum() / max(fa["size"].sum(), 1)
                rows.append({"matcher": name, "class": cls, "group_by": g, "group": grp,
                             "n_persons": len(ids[grp]), "n_nonmated": len(a), "n_mated": len(b),
                             "FMR": fmr, "FMR_lo": np.nanquantile(bf, .025), "FMR_hi": np.nanquantile(bf, .975),
                             "FMR_ratio": fmr / cls_fmr if cls_fmr > 0 else np.nan,
                             "FMR_ratio_lo": np.nanquantile(bf, .025) / cls_fmr if cls_fmr > 0 else np.nan,
                             "FMR_ratio_hi": np.nanquantile(bf, .975) / cls_fmr if cls_fmr > 0 else np.nan,
                             "FNMR": fb["sum"].sum() / max(fb["size"].sum(), 1),
                             "FNMR_lo": np.nanquantile(bn, .025), "FNMR_hi": np.nanquantile(bn, .975),
                             "FMR_bal": np.nanmean(bal_f), "FNMR_bal": np.nanmean(bal_n)})
        log(f"{name}: FMR {cls} done")
    return pd.DataFrame(rows)


def lr_part(p, name, B, repeats, rng, transform="none"):
    from analyze_by_condition import CLASSES as SEL
    from facebias.lr import cross_validated_llrs
    rows = []
    p = p.dropna(subset=["score"])
    for cls in CLASSES:
        d = p[SEL[cls](p.condition_b.astype(str)) & p.sd].reset_index(drop=True)
        if d.mated.nunique() < 2:
            continue
        d["llr_g"] = cross_validated_llrs(d.score, d.mated, d.id_a, d.id_b, n_repeats=repeats,
                                          calibrator="logit", transform=transform)
        for g in GROUPS:
            col = f"{g}_a"
            d["llr_s"] = np.nan
            for grp, ix in d.groupby(col).groups.items():
                dg = d.loc[ix]
                if dg.mated.nunique() < 2 or dg.mated.sum() < 20:
                    continue
                d.loc[ix, "llr_s"] = cross_validated_llrs(dg.score, dg.mated, dg.id_a, dg.id_b, n_repeats=repeats,
                                                          calibrator="logit", transform=transform)
            v = d.dropna(subset=["llr_g", "llr_s"]).assign(diff=lambda x: x.llr_g - x.llr_s)
            groups = sorted(v[col].unique())
            # per person sums for the joint bootstrap (resample persons within each group)
            stats = {}
            for kind, mval in (("mated", 1), ("nonmated", 0)):
                bg = []
                for grp in groups:
                    w = v[(v[col] == grp) & (v.mated == mval)].groupby("id_a")["diff"].agg(["sum", "size"])
                    bg.append(boot_ratio(w["sum"].values, w["size"].values.astype(float), B, rng))
                    stats.setdefault(grp, {})[kind] = w["sum"].sum() / max(w["size"].sum(), 1)
                BG = np.vstack(bg)                                   # groups x B
                rel = BG - np.nanmean(BG, axis=0, keepdims=True)
                for i, grp in enumerate(groups):
                    stats[grp][f"{kind}_lo"], stats[grp][f"{kind}_hi"] = np.nanquantile(BG[i], [.025, .975])
                    stats[grp][f"{kind}_rel_lo"], stats[grp][f"{kind}_rel_hi"] = np.nanquantile(rel[i], [.025, .975])
            mean_m = np.mean([stats[g_]["mated"] for g_ in groups]); mean_n = np.mean([stats[g_]["nonmated"] for g_ in groups])
            for grp in groups:
                s, vv = stats[grp], v[v[col] == grp]
                n_, m_ = vv[vv.mated == 0], vv[vv.mated == 1]
                rows.append({"matcher": name, "class": cls, "group_by": g, "group": grp,
                             "bias_mated": s["mated"], "bias_mated_lo": s["mated_lo"], "bias_mated_hi": s["mated_hi"],
                             "bias_mated_rel": s["mated"] - mean_m, "bias_mated_rel_lo": s["mated_rel_lo"],
                             "bias_mated_rel_hi": s["mated_rel_hi"],
                             "bias_nonmated": s["nonmated"], "bias_nonmated_rel": s["nonmated"] - mean_n,
                             "bias_nonmated_rel_lo": s["nonmated_rel_lo"], "bias_nonmated_rel_hi": s["nonmated_rel_hi"],
                             "RMED_global": (n_.llr_g > 0).mean(), "RMED_group": (n_.llr_s > 0).mean(),
                             "RMEP_global": (m_.llr_g < 0).mean(), "RMEP_group": (m_.llr_s < 0).mean()})
        log(f"{name}: LR {cls} done")
    return pd.DataFrame(rows)


def ci(v, lo, hi, d=2, pct=False):
    f = (lambda x: f"{100 * x:.{d}f}") if pct else (lambda x: f"{x:.{d}f}")
    if not np.isfinite(v):
        return ""
    return f"{f(v)} [{f(lo)}; {f(hi)}]"


def summary(models):
    md = ["# Precieze resultaten met 95%-betrouwbaarheidsintervallen\n",
          "Intervallen: bootstrap over referentiepersonen (identiteit-geclusterd). FMR in %, drempel per conditieklasse "
          "bij FMR = 0,1% over alle paren van die klasse. FMR-ratio = FMR van de groep / FMR van de klasse. "
          "LR-bias = gemiddelde log10 LR (globaal − groepsspecifiek), relatief t.o.v. het gemiddelde over de groepen; "
          "> 0: globale LR's te hoog voor die groep. Een interval dat 0 (bias) of 1 (ratio) niet bevat, is een "
          "duidelijk verschil.\n",
          "Let op: de intervallen van de LR-bias geven alleen de steekproefonzekerheid over personen bij vaste, "
          "gefitte kalibraties (10 CV-herhalingen); de onzekerheid van de kalibratie zelf zit er niet in, dus ze "
          "zijn eerder te smal dan te breed. De FMR-intervallen zijn volledig.\n"]
    for m in models:
        f, l = OUT / f"{m}_fmr.csv", OUT / f"{m}_lr.csv"
        if not f.exists():
            continue
        F, L = pd.read_csv(f), pd.read_csv(l) if l.exists() else pd.DataFrame()
        md.append(f"\n## {m}\n")
        for cls in ["edit_hq", "cctv_still", "video_P"]:
            for g in GROUPS:
                t = F[(F["class"] == cls) & (F.group_by == g)].set_index("group")
                if t.empty:
                    continue
                out = pd.DataFrame({
                    "FMR % [95%-BI]": [ci(r.FMR, r.FMR_lo, r.FMR_hi, 3, True) for r in t.itertuples()],
                    "FMR-ratio [95%-BI]": [ci(r.FMR_ratio, r.FMR_ratio_lo, r.FMR_ratio_hi, 2) for r in t.itertuples()],
                    "FNMR % [95%-BI]": [ci(r.FNMR, r.FNMR_lo, r.FNMR_hi, 1, True) for r in t.itertuples()]},
                    index=t.index)
                if len(L):
                    u = L[(L["class"] == cls) & (L.group_by == g)].set_index("group")
                    out["LR-bias mated rel [95%-BI]"] = [ci(r.bias_mated_rel, r.bias_mated_rel_lo, r.bias_mated_rel_hi)
                                                         if r.Index in u.index else "" for r in u.reindex(t.index).itertuples()]
                    out["LR-bias non-mated rel [95%-BI]"] = [ci(r.bias_nonmated_rel, r.bias_nonmated_rel_lo, r.bias_nonmated_rel_hi)
                                                             for r in u.reindex(t.index).itertuples()]
                out = out.loc[t.FMR_ratio.sort_values(ascending=False).index]
                md += [f"\n### {cls} · {g}\n", out.to_markdown() if hasattr(out, "to_markdown") else out.to_string()]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    forest(models)
    log("-> results_precise/summary.md")


def forest(models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cls in ["cctv_still", "video_P"]:
        frames = []
        for m in models:
            f = OUT / f"{m}_fmr.csv"
            if f.exists():
                F = pd.read_csv(f)
                frames.append(F[(F["class"] == cls) & (F.group_by == "ancestry")])
        if not frames:
            continue
        A = pd.concat(frames)
        order = A.groupby("group").FMR_ratio.mean().sort_values().index.tolist()
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.5))
        ms = A.matcher.unique()
        for k, m in enumerate(ms):
            t = A[A.matcher == m].set_index("group").reindex(order)
            y = np.arange(len(order)) + (k - len(ms) / 2) * 0.12
            ax.errorbar(t.FMR_ratio, y, xerr=[t.FMR_ratio - t.FMR_ratio_lo, t.FMR_ratio_hi - t.FMR_ratio],
                        fmt="o", ms=3, lw=0.8, label=m)
        ax.axvline(1, color="k", lw=0.8); ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel("FMR-ratio t.o.v. de hele klasse (95%-BI)"); ax.set_title(f"{cls}: FMR per herkomst")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(OUT / f"forest_{cls}.png", dpi=130)


def main(argv=None):
    from facebias.embed import MATCHERS
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MATCHERS))
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--transform", default=None, help="none (cosine) or logit (scores in [0,1], e.g. System C); "
                                                     "default: logit for system_c, none otherwise")
    a = p.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for m in a.models:
        src = ROOT / f"data/scored_{m}_v2.csv"
        if not src.exists():
            log(f"{m}: {src.name} missing, skipped"); continue
        p_ = pd.read_csv(src, usecols=["id_a", "id_b", "mated", "condition_b", "score", "same_demo",
                                       "ancestry_a", "gender_presentation_a", "age_band_a"])
        p_["sd"] = p_.same_demo.astype(str).str.lower().eq("true")
        if not (OUT / f"{m}_fmr.csv").exists():
            fmr_part(p_, m, a.boot, rng).to_csv(OUT / f"{m}_fmr.csv", index=False)
        if not (OUT / f"{m}_lr.csv").exists():
            lr_part(p_, m, a.boot, a.repeats, rng, a.transform or ("logit" if "system_c" in m else "none")).to_csv(OUT / f"{m}_lr.csv", index=False)
    summary(a.models)


if __name__ == "__main__":
    main()
