"""
False match rate per group on REAL faces (FairFace train split, adults), open-source recognisers,
with the same design as the System C run on the System C workstation (system_c_run/fairface/FAIRFACE_TRAIN_RESULTATEN.md):
  gallery : 357 images per ancestry x gender (seeded), probes: all adult train faces
  pairs   : probe vs gallery of the same ancestry and gender, own image excluded
  dedup   : pairs scoring >= --dedup (cosine) are treated as the same photo/person and left out
  threshold: the score where the mean of the per-group FMRs is 1e-3 ("balanced over the groups")
FairFace has no identity labels: every remaining pair counts as non-mated. Only group averages are
written (FairFace stays local, no images or per-image results leave this computer).

    python tools/fairface_fmr.py --models adaface adaface_kprpe arcface
Output: results_extended/fairface_fmr.md / .csv
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

OUT = ROOT / "results_extended"
ADULT = {"20-29", "30-39", "40-49", "50-59", "60-69", "70+"}


def load_train(cache: Path):
    metas, crops = [], []
    for k in (1, 2):                       # parts 1 and 2 = FairFace train split
        z = np.load(cache / f"fairface_crops_part{k}.npz", allow_pickle=True)
        m = pd.DataFrame({"race": z["race"], "gender": z["gender"], "age": z["age"]})
        keep = z["ok"] & m.age.isin(ADULT).values
        metas.append(m[keep]); crops.append(z["crops"][keep])
    return pd.concat(metas, ignore_index=True), np.concatenate(crops)


def scores_by_stratum(E, meta, gal_idx):
    """For every (race, gender): cosine of all probes of that stratum vs its gallery (own image -> nan)."""
    out = {}
    for (r, g), pr in meta.groupby(["race", "gender"]).groups.items():
        gi = np.array([i for i in gal_idx if meta.race.iat[i] == r and meta.gender.iat[i] == g])
        pi = np.array(list(pr))
        S = E[pi] @ E[gi].T
        S[pi[:, None] == gi[None, :]] = np.nan
        out[(r, g)] = (pi, S)
    return out


def fmr_at(strata, t, dedup):
    rows = []
    for (r, g), (pi, S) in strata.items():
        v = S[np.isfinite(S) & (S < dedup)]
        rows.append({"race": r, "gender": g, "n": v.size, "fm": int((v >= t).sum())})
    return pd.DataFrame(rows)


def balanced_threshold(strata, dedup, target=1e-3):
    lo, hi = -50.0, dedup
    for _ in range(60):
        t = (lo + hi) / 2
        d = fmr_at(strata, t, dedup).groupby("race")[["n", "fm"]].sum()
        m = (d.fm / d.n).mean()
        lo, hi = (t, hi) if m > target else (lo, t)
    return (lo + hi) / 2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["adaface", "adaface_kprpe", "arcface"])
    p.add_argument("--cache", default="data/cache_extended")
    p.add_argument("--per-stratum", type=int, default=357)
    p.add_argument("--dedup", type=float, default=0.75, help="cosine above which a pair is taken as the same photo/person")
    a = p.parse_args(argv)
    meta, crops = load_train(Path(a.cache))
    fc.log(f"FairFace train adults: {len(meta)} faces")
    rng = np.random.default_rng(0)
    gal = []
    for _, idx in meta.groupby(["race", "gender"]).groups.items():
        idx = np.array(list(idx))
        gal += list(rng.choice(idx, size=min(a.per_stratum, len(idx)), replace=False))
    gal = np.array(sorted(gal))
    rows, srows = [], []
    for name in a.models:
        order = "bgr"
        f = ROOT / "data/embeddings_v2" / f"{name}_channel_order.json"
        if f.exists():
            order = json.loads(f.read_text())["channel_order"]
        E = fc.embed(name, crops, np.ones(len(crops), bool), order)
        strata = scores_by_stratum(E, meta, gal)
        t = balanced_threshold(strata, a.dedup)
        d = fmr_at(strata, t, a.dedup)
        byr = d.groupby("race")[["n", "fm"]].sum(); byr["FMR"] = byr.fm / byr.n
        med = byr.FMR.median()
        for r, x in byr.iterrows():
            sub = d[d.race == r].set_index("gender")
            rows.append({"matcher": name, "group": r, "pairs": int(x.n), "FMR": x.FMR, "rel_median": x.FMR / med,
                         "FMR_man": sub.fm.get("man", 0) / max(sub.n.get("man", 1), 1),
                         "FMR_woman": sub.fm.get("woman", 0) / max(sub.n.get("woman", 1), 1)})
        byg = d.groupby("gender")[["n", "fm"]].sum()
        srows.append({"matcher": name, "threshold": t, "max/min": byr.FMR.max() / byr.FMR.min(),
                      "women/men": (byg.fm / byg.n).get("woman") / (byg.fm / byg.n).get("man"),
                      "order": " > ".join(byr.FMR.sort_values(ascending=False).index)})
        fc.log(f"{name}: threshold {t:.3f}, max/min {byr.FMR.max() / byr.FMR.min():.1f}")
        del E, strata
    R, S = pd.DataFrame(rows), pd.DataFrame(srows)
    R.to_csv(OUT / "fairface_fmr.csv", index=False); S.to_csv(OUT / "fairface_fmr_summary.csv", index=False)
    piv = R.pivot_table(index="group", columns="matcher", values="rel_median")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    md = ["# FMR per group on real faces (FairFace train, adults), open-source recognisers\n",
          f"Same design as the System C run: {a.per_stratum} gallery images per ancestry x gender, probes = all adults, "
          f"same ancestry and gender, own image excluded, pairs with cosine >= {a.dedup} left out as duplicates, "
          "threshold where the mean group FMR is 1e-3.\n",
          "## FMR relative to the median group\n", piv.round(2).to_string(), "\n## Summary\n", S.round(3).to_string(index=False)]
    (OUT / "fairface_fmr.md").write_text("\n".join(md), encoding="utf-8")
    fc.log("-> results_extended/fairface_fmr.md")


if __name__ == "__main__":
    main()
