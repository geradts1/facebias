"""
Real faces (FairFace train, adults), two questions the synthetic set cannot answer on its own:

1. Does CCTV-like image quality change the group differences on REAL faces?
   Gallery = original photos (357 per ancestry x gender), probes = every adult face at three
   qualities: original, cctv_mid (face ~64 px, blur, JPEG 55, noise) and cctv_low (~32 px, JPEG 35),
   the same degradations as the synthetic CCTV stills, applied to the aligned crop.
2. Does top-100 T-norm (the label-free mitigation) also shrink the gap on real faces?
   Cohort = 600 random adult faces not in the gallery (mixed, no labels used); each probe score is
   normalised with the mean/sd of that probe's 100 highest cohort scores.

Non-mated pairs only (FairFace has no identity labels): same ancestry and gender, own image excluded,
pairs whose ORIGINAL images score cosine >= --dedup treated as the same person and left out.
Threshold per condition and method: the score where the mean group FMR is 1e-3.
Only group averages are written (FairFace stays on this computer).

    python tools/fairface_real2.py --models adaface adaface_kprpe
Output: results_extended/fairface_real2.md / .csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import fairface_compare as fc  # noqa: E402
from fairface_fmr import load_train, balanced_threshold, fmr_at  # noqa: E402

OUT = ROOT / "results_extended"
DEGR = {"original": None,
        "cctv_mid": {"face_px": 64, "blur": 0.6, "jpeg": 55, "noise": 5},
        "cctv_low": {"face_px": 32, "blur": 1.0, "jpeg": 35, "noise": 8}}
FACE_W = 80.0   # approximate face width (px) inside the 112x112 aligned crop


def degrade_all(crops, p, seed=0):
    from PIL import Image
    from facebias.generate import degrade
    if p is None:
        return crops
    rng = np.random.default_rng(seed)
    out = np.empty_like(crops)
    for i, c in enumerate(crops):
        im = Image.fromarray(c[..., ::-1])                       # BGR -> RGB
        d = degrade(im, FACE_W, rng=rng, **p)
        out[i] = np.asarray(d)[..., ::-1]
    return out


def strata_scores(Ep, Eg, meta, gal, valid_pair):
    out = {}
    for (r, g), pr in meta.groupby(["race", "gender"]).groups.items():
        gi = gal[(meta.race.values[gal] == r) & (meta.gender.values[gal] == g)]
        pi = np.array(list(pr))
        S = Ep[pi] @ Eg[gi].T
        S[~valid_pair[(r, g)]] = np.nan
        out[(r, g)] = (pi, S)
    return out


def tnorm_top(Ep, Ec, k=100):
    C = Ep @ Ec.T
    top = -np.sort(-C, axis=1)[:, :k]
    return top.mean(1), np.maximum(top.std(1), 1e-6)


def boot_ci(st, t, races, B=200, seed=0):
    """95% CI of each group's FMR relative to the median group and of max/min, bootstrap over probe images."""
    rng = np.random.default_rng(seed)
    per = {}   # race -> (n per probe, fm per probe)
    for (r, g), (pi, S) in st.items():
        ok = np.isfinite(S)
        n, fm = ok.sum(1), ((S >= t) & ok).sum(1)
        a, b = per.get(r, (np.zeros(0), np.zeros(0)))
        per[r] = (np.concatenate([a, n]), np.concatenate([b, fm]))
    rel, mm = {r: [] for r in races}, []
    for _ in range(B):
        f = {}
        for r in races:
            n, fm = per[r]
            i = rng.integers(0, len(n), len(n))
            f[r] = fm[i].sum() / max(n[i].sum(), 1)
        v = pd.Series(f); med = v.median()
        for r in races:
            rel[r].append(v[r] / med if med > 0 else np.nan)
        mm.append(v.max() / v.min() if v.min() > 0 else np.nan)
    q = lambda x: (np.nanquantile(x, 0.025), np.nanquantile(x, 0.975))
    return {r: q(rel[r]) for r in races}, q(mm)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["adaface", "adaface_kprpe"])
    p.add_argument("--cache", default="data/cache_extended")
    p.add_argument("--per-stratum", type=int, default=357)
    p.add_argument("--cohort", type=int, default=600)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--dedup", type=float, default=0.75)
    p.add_argument("--tag", default="", help="suffix for the output files")
    p.add_argument("--boot", type=int, default=200)
    a = p.parse_args(argv)
    meta, crops = load_train(Path(a.cache))
    meta = meta.reset_index(drop=True)
    fc.log(f"FairFace train adults: {len(meta)}")
    rng = np.random.default_rng(0)
    gal = []
    for _, idx in meta.groupby(["race", "gender"]).groups.items():
        idx = np.array(list(idx))
        gal += list(rng.choice(idx, size=min(a.per_stratum, len(idx)), replace=False))
    gal = np.array(sorted(gal))
    rest = np.setdiff1d(np.arange(len(meta)), gal)
    cohort = rng.choice(rest, size=a.cohort, replace=False)
    rows, srows = [], []
    for name in a.models:
        order = "bgr"
        f = ROOT / "data/embeddings_v2" / f"{name}_channel_order.json"
        if f.exists():
            order = json.loads(f.read_text())["channel_order"]
        E = {}
        for c, prm in DEGR.items():          # degrade one condition at a time (memory)
            cr = degrade_all(crops, prm, seed=1)
            E[c] = fc.embed(name, cr, np.ones(len(cr), bool), order)
            del cr
            fc.log(f"{name}: embedded {c}")
        Eo = E["original"]
        # valid non-mated pairs: own image out, near-duplicates (on the original photos) out
        valid = {}
        for (r, g), pr in meta.groupby(["race", "gender"]).groups.items():
            gi = gal[(meta.race.values[gal] == r) & (meta.gender.values[gal] == g)]
            pi = np.array(list(pr))
            valid[(r, g)] = (pi[:, None] != gi[None, :]) & ((Eo[pi] @ Eo[gi].T) < a.dedup)
        for cond in DEGR:
            raw = strata_scores(E[cond], Eo, meta, gal, valid)
            mu, sd = tnorm_top(E[cond], Eo[cohort], a.topk)
            norm = {k: (pi, (S - mu[pi, None]) / sd[pi, None]) for k, (pi, S) in raw.items()}
            C = E[cond] @ Eo[cohort].T
            mu_a, sd_a = C.mean(1), np.maximum(C.std(1), 1e-6)
            del C
            tn = {k: (pi, (S - mu_a[pi, None]) / sd_a[pi, None]) for k, (pi, S) in raw.items()}
            for method, st in [("raw", raw), ("T-norm (whole cohort)", tn), ("top-100 T-norm", norm)]:
                cap = 2.0 if method == "raw" else 1e3    # pairs already de-duplicated via `valid`
                t = balanced_threshold(st, cap)
                d = fmr_at(st, t, cap)
                byr = d.groupby("race")[["n", "fm"]].sum(); byr["FMR"] = byr.fm / byr.n
                byg = d.groupby("gender")[["n", "fm"]].sum(); byg["FMR"] = byg.fm / byg.n
                med = byr.FMR.median()
                ci, mmci = boot_ci(st, t, list(byr.index), B=a.boot)
                for r, x in byr.iterrows():
                    rows.append({"matcher": name, "condition": cond, "method": method, "group": r,
                                 "FMR": x.FMR, "rel_median": x.FMR / med if med > 0 else np.nan,
                                 "rel_lo": ci[r][0], "rel_hi": ci[r][1]})
                srows.append({"matcher": name, "condition": cond, "method": method,
                              "max/min": byr.FMR.max() / byr.FMR.min() if byr.FMR.min() > 0 else np.inf,
                              "max/min_lo": mmci[0], "max/min_hi": mmci[1],
                              "max/median": byr.FMR.max() / med if med > 0 else np.inf,
                              "women/men": byg.FMR.get("woman") / byg.FMR.get("man"),
                              "top": byr.FMR.idxmax(), "bottom": byr.FMR.idxmin()})
            fc.log(f"{name}: {cond} done")
        del E
    R, S = pd.DataFrame(rows), pd.DataFrame(srows)
    R.to_csv(OUT / f"fairface_real2{a.tag}.csv", index=False); S.to_csv(OUT / f"fairface_real2{a.tag}_summary.csv", index=False)
    md = ["# Real faces (FairFace train): CCTV degradation and top-100 T-norm\n",
          "Gallery original photos, probes at three qualities; non-mated pairs of the same ancestry and gender; "
          "threshold where the mean group FMR is 1e-3. Only group averages.\n",
          "## Summary\n", S.round(2).to_string(index=False)]
    for name in a.models:
        t = R[(R.matcher == name)].pivot_table(index="group", columns=["condition", "method"], values="rel_median")
        md += [f"\n## {name}: FMR relative to the median group\n", t.round(2).to_string()]
    (OUT / f"fairface_real2{a.tag}.md").write_text("\n".join(md), encoding="utf-8")
    fc.log("-> results_extended/fairface_real2.md")


if __name__ == "__main__":
    main()
