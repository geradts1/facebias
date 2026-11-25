"""
Bias and LR analysis per probe condition (reference photo vs ...), for the open-source
matchers (and later System C, same input format).

    python tools/analyze_by_condition.py                    # adaface + arcface, pass-1 detections
    python tools/analyze_by_condition.py --rescued          # also faces found by the 2nd detector
    python tools/analyze_by_condition.py --scored data/scored_system_c.csv --name system_c

Condition classes (probe side; the A side is always the reference photo):
  passfoto   ref2                               second passport photo
  edit_hq    hq                                 situation edits, full quality
  cctv_still cctv_mid, cctv_low                 edits with CCTV degradation
  video_I    *_I                                video stills, I-frame
  video_P    *_P                                video stills, last P-frame

Per class and per group (ancestry, gender, age band):
  - FMR/FNMR at a class-specific threshold (FMR = 1e-3 over all pairs of that class), d'
  - the same on a *balanced* subsample (equal number of reference identities per group,
    5 random draws, mean) so group sizes do not drive the comparison
  - LR level: Cllr_cal of a globally calibrated system vs a group-specific one
  - FTE % (pairs without a score because no face was found)
Output: results_by_condition_<name>.md and .csv
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

from facebias.bias import dprime, error_rates, lr_level_bias, threshold_at_fmr  # noqa: E402

CLASSES = {
    "passfoto": lambda c: c == "ref2",
    "edit_hq": lambda c: c == "hq",
    "cctv_still": lambda c: c.isin(["cctv_mid", "cctv_low"]),
    "video_I": lambda c: c.str.endswith("_I"),
    "video_P": lambda c: c.str.endswith("_P"),
}
GROUPS = ["ancestry", "gender_presentation", "age_band"]


def rescore_with_rescued(model):
    """Recompute cosine scores using pass-1 embeddings plus faces rescued by YuNet."""
    from run_open_matchers import EMB, EXPORT, load_embeddings
    idx, E, ok = load_embeddings(model)
    emb = {f: E[i] for f, i in idx.items() if ok[i]}
    z = np.load(EMB / f"{model}_det2.npz", allow_pickle=True)
    for f, e in zip(z["files"], z["emb"].astype(np.float32)):
        emb.setdefault(f, e)
    pairs = pd.read_csv(EXPORT / "pairs_meta.csv")
    name = lambda p: Path(str(p).replace("\\", "/")).name
    fa, fb = pairs.image_a.map(name), pairs.image_b.map(name)
    s = np.full(len(pairs), np.nan)
    for k, (a, b) in enumerate(zip(fa, fb)):
        ea, eb = emb.get(a), emb.get(b)
        if ea is not None and eb is not None:
            s[k] = float(np.dot(ea, eb))
    pairs["score"], pairs["fte"] = s, np.isnan(s)
    out = ROOT / f"data/scored_{model}_rescued.csv"
    pairs.to_csv(out, index=False)
    print(f"{model}: rescored with rescued faces -> {out}; FTE pairs {pairs.fte.mean():.2%}", flush=True)
    return pairs


def balanced_rates(d, group_col, thr, seeds=5):
    """Equal number of reference identities per group; mean FMR/FNMR over draws."""
    ids = d.groupby(f"{group_col}_a").id_a.unique()
    n = min(len(v) for v in ids)
    out = {}
    for g, v in ids.items():
        fm, fn = [], []
        for s in range(seeds):
            pick = np.random.default_rng(s).choice(v, n, replace=False)
            sub = d[d.id_a.isin(pick)]
            r = error_rates(sub.score, sub.mated, thr)
            fm.append(r[0]); fn.append(r[1])
        out[g] = (float(np.nanmean(fm)), float(np.nanmean(fn)), n)
    return out


def analyse(pairs, name, do_lr=True):
    rows, md = [], [f"# Bias per conditie: {name}\n",
                    "A-kant is altijd de referentiefoto. Drempel per conditieklasse bij FMR = 1e-3 (over alle paren "
                    "van die klasse). 'bal' = gebalanceerde steekproef (gelijk aantal identiteiten per groep, "
                    "gemiddelde van 5 trekkingen). Cllr_cal glob/spec = kalibratieverlies van één globaal LR-model "
                    "vs een model per groep (hoger glob dan spec = globale LR's zitten voor die groep scheef).\n"]
    pairs = pairs.copy()
    pairs["same_demo"] = pairs.same_demo.astype(str).str.lower().eq("true")
    for cls, sel in CLASSES.items():
        part = pairs[sel(pairs.condition_b.astype(str))]
        if part.empty:
            continue
        fte = part.score.isna()
        d = part[~fte]
        thr = threshold_at_fmr(d.score, d.mated, 1e-3)
        fmr, fnmr = error_rates(d.score, d.mated, thr)
        md.append(f"\n## {cls}\n\n{len(part)} paren, FTE {fte.mean():.1%}; drempel {thr:.3f}: "
                  f"FMR {fmr:.4f}, FNMR {fnmr:.3f}, d' {dprime(d.score, d.mated):.2f}\n")
        for g in GROUPS:
            same = d[d.same_demo]
            t = []
            bal = balanced_rates(same, g, thr)
            lr = None
            if do_lr and same.mated.nunique() == 2:
                try:
                    lr, _ = lr_level_bias(same, g, n_boot=0, calibrator="logit", transform="none")
                except Exception as e:   # small cells
                    print(f"  LR {cls}/{g} skipped: {e}")
            for grp, dg in same.groupby(f"{g}_a"):
                fm, fn = error_rates(dg.score, dg.mated, thr)
                allg = part[part[f"{g}_a"] == grp]
                row = {"model": name, "class": cls, "group_by": g, "group": grp,
                       "n_mated": int(dg.mated.sum()), "FMR": fm, "FNMR": fn, "d'": dprime(dg.score, dg.mated),
                       "FMR_bal": bal[grp][0], "FNMR_bal": bal[grp][1],
                       "FTE_%": 100 * allg.score.isna().mean()}
                if lr is not None and grp in lr.index:
                    row["Cllr_cal_glob"] = lr.loc[grp].get("global_Cllr_cal")
                    row["Cllr_cal_spec"] = lr.loc[grp].get("specific_Cllr_cal")
                rows.append(row); t.append(row)
            tab = pd.DataFrame(t).set_index("group").drop(columns=["model", "class", "group_by"])
            fm_ = tab.FMR_bal.replace(0, np.nan)
            md.append(f"\n### {cls} · {g}  (FMR_bal max/min {fm_.max() / fm_.min():.1f}, "
                      f"FNMR_bal max/min {tab.FNMR_bal.max() / max(tab.FNMR_bal.min(), 1e-9):.2f})\n")
            try:
                md.append(tab.round(4).to_markdown())
            except ImportError:
                md.append("```\n" + tab.round(4).to_string() + "\n```")
        print(f"{name}: {cls} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / f"results_by_condition_{name}.csv", index=False)
    (ROOT / f"results_by_condition_{name}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"-> results_by_condition_{name}.md", flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["adaface", "arcface"])
    p.add_argument("--rescued", action="store_true")
    p.add_argument("--scored", help="other scored pairs CSV (e.g. System C import)")
    p.add_argument("--name")
    p.add_argument("--no-lr", action="store_true")
    a = p.parse_args(argv)
    if a.scored:
        analyse(pd.read_csv(a.scored), a.name or Path(a.scored).stem, not a.no_lr)
        return
    for m in a.models:
        pairs = rescore_with_rescued(m) if a.rescued else pd.read_csv(ROOT / f"data/scored_{m}.csv")
        analyse(pairs, m + ("_rescued" if a.rescued else ""), not a.no_lr)


if __name__ == "__main__":
    main()
