"""
Real faces WITH identity labels (RFW, BFW, BUPT-Balancedface or any set laid out as
<root>/<group>/<identity>/<image>): the test the synthetic set and FairFace cannot give,
namely MATED pairs per demographic group, so LR calibration can be checked on real faces.

Design (same as the synthetic study, casework-like):
  reference : first image of each identity, original quality
  probes    : the other images of that identity (max --per-id), at three qualities
              original / cctv_mid (~64 px, blur, JPEG 55) / cctv_low (~32 px, JPEG 35)
  mated     : reference vs own probes
  non-mated : reference vs probes of up to --k other identities of the same group
Score normalisation: raw; T-norm and top-100 T-norm against a cohort of 600 SYNTHETIC
references (no labels, no real data in the cohort - the same cohort idea as on SCface).
Per matcher x condition x normalisation:
  * threshold at FMR 1e-3 over all non-mated pairs -> FMR and FNMR per group (bootstrap over identities)
  * LR: logistic + ELUB, identity-disjoint 5-fold CV (3 repeats), global vs per group
        -> Cllr / Cllr_min per group, relative LR shift of mated pairs, spread
  * transfer: calibration fitted on the SYNTHETIC set (results_extended/synth_cal_<matcher>_<cls>.json if
    present) applied to the real scores -> Cllr per group (does synthetic calibration carry over?)
Only group-level aggregates are written; no images or per-image scores leave this computer.

    python tools/real_mated.py --root data/real/RFW/data --name rfw
    python tools/real_mated.py --root data/real/bfw --name bfw   (group folders like asian_females)
Output: results_extended/real_mated_<name>.md / .csv
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
from fairface_real2 import DEGR, degrade_all  # noqa: E402
from facebias.lr import cross_validated_llrs, cllr, cllr_min, ScoreLRSystem  # noqa: E402

OUT = ROOT / "results_extended"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def manifest(root: Path, per_id: int, max_ids: int, seed=0):
    rows = []
    for gdir in sorted(p for p in root.iterdir() if p.is_dir()):
        ids = sorted(p for p in gdir.iterdir() if p.is_dir())
        rng = np.random.default_rng(seed)
        if max_ids and len(ids) > max_ids:
            ids = list(rng.choice(ids, max_ids, replace=False))
        for idir in ids:
            imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT)
            if len(imgs) < 2:
                continue
            for k, p in enumerate(imgs[:per_id + 1]):
                rows.append({"path": str(p), "group": gdir.name, "identity": f"{gdir.name}/{idir.name}", "ref": k == 0})
    m = pd.DataFrame(rows)
    g = m.group.str.lower()
    m["gender"] = np.where(g.str.endswith("females"), "woman", np.where(g.str.endswith("males"), "man", ""))
    m["ancestry"] = g.str.replace("_females", "").str.replace("_males", "").str.capitalize()
    return m


def load_crops(m: pd.DataFrame, cache: Path):
    f = cache / "crops.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if len(z["ok"]) == len(m):
            return z["crops"], z["ok"]
    import cv2
    imgs = [cv2.imread(p) for p in m.path]
    crops, ok = fc.align_all(imgs)
    ok0 = ok.copy()
    # pre-cropped sets (BFW): if no face is found, fall back to a plain resize of the crop
    for i in np.where(~ok)[0]:
        if imgs[i] is not None:
            crops[i] = cv2.resize(imgs[i], (112, 112)); ok[i] = True
    fc.log(f"face found and aligned: {int(len(ok) - (~ok0).sum())}/{len(ok)}; plain resize for the rest")
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, crops=crops, ok=ok)
    return crops, ok


def make_pairs(m, k, seed=0):
    rng = np.random.default_rng(seed)
    ref = m[m.ref].set_index("identity")
    pro = m[~m.ref]
    by_id = {i: d.index.values for i, d in pro.groupby("identity")}
    rows = []
    for (grp, gen), d in ref.groupby(["group", "gender"]):
        ids = [i for i in d.index if i in by_id]
        for i in ids:
            ri = m.index[(m.identity == i) & m.ref][0]
            rows += [(ri, j, i, i, 1) for j in by_id[i]]
            others = [o for o in ids if o != i]
            for o in rng.choice(others, min(k, len(others)), replace=False) if others else []:
                rows += [(ri, j, i, o, 0) for j in by_id[o]]
    P = pd.DataFrame(rows, columns=["ia", "ib", "id_a", "id_b", "mated"])
    P["ancestry"] = P.ia.map(m.ancestry); P["gender"] = P.ia.map(m.gender)
    return P


def synth_cohort(n=600, seed=0):
    meta, crops, ok = fc.load_synth_from_cache(Path("data/cache_extended"))
    refs = np.where(meta.kind.values == "reference")[0]
    sel = np.random.default_rng(seed).choice(refs, min(n, len(refs)), replace=False)
    return crops[sel]


def boot_rate(hit, ids, B=500, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ids, return_inverse=True)
    s, n = np.bincount(inv, hit), np.bincount(inv)
    v = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(u), len(u)) for _ in range(B))]
    return np.quantile(v, [0.025, 0.975])


def analyse(P, sc, name, cond, method, rows, srows, synth_cal=None):
    d = P.assign(score=sc).dropna(subset=["score"]).reset_index(drop=True)
    y = d.mated.values
    nm = d[d.mated == 0]
    thr = np.quantile(nm.score, 1 - 1e-3)
    fmr_all = (nm.score >= thr).mean()
    kw = dict(transform="none")          # cosine in [-1, 1] and T-norm z-scores: no logit
    g_llr = cross_validated_llrs(d.score.values, y, d.id_a.values, d.id_b.values, **kw)
    d["llr"] = g_llr
    t_llr = None
    if synth_cal is not None:
        t_llr = synth_cal.llr(d.score.values)
    shifts = {}
    for g, dg in d.groupby("ancestry"):
        s_llr = cross_validated_llrs(dg.score.values, dg.mated.values, dg.id_a.values, dg.id_b.values, **kw)
        ok = np.isfinite(s_llr) & np.isfinite(dg.llr.values) & (dg.mated.values == 1)
        shifts[g] = float(np.mean(dg.llr.values[ok] - s_llr[ok])) if ok.any() else np.nan
    mean_shift = np.nanmean(list(shifts.values()))
    for g, dg in d.groupby("ancestry"):
        ok = np.isfinite(dg.llr.values)
        m_, n_ = dg[dg.mated == 1], dg[dg.mated == 0]
        fm = (n_.score >= thr).values.astype(float); fn = (m_.score < thr).values.astype(float)
        r = {"matcher": name, "condition": cond, "method": method, "group": g,
             "n_ids": dg.id_a.nunique(), "n_mated": len(m_), "n_nonmated": len(n_),
             "FMR": fm.mean(), "FMR_ratio": fm.mean() / fmr_all if fmr_all > 0 else np.nan,
             "FMR_lo": boot_rate(fm, n_.id_a.values)[0], "FMR_hi": boot_rate(fm, n_.id_a.values)[1],
             "FNMR": fn.mean(),
             "Cllr": cllr(dg.llr.values[ok], dg.mated.values[ok]),
             "Cllr_min": cllr_min(dg.llr.values[ok], dg.mated.values[ok]),
             "LR_shift_rel": shifts[g] - mean_shift}
        if t_llr is not None:
            tl = t_llr[dg.index.values]
            r["Cllr_synthcal"] = cllr(tl, dg.mated.values)
            r["misl_nm_LR>1"] = float((tl[dg.mated.values == 0] > 0).mean())
            r["misl_m_LR<1"] = float((tl[dg.mated.values == 1] < 0).mean())
        rows.append(r)
    G = pd.DataFrame([r for r in rows if r["matcher"] == name and r["condition"] == cond and r["method"] == method])
    ok = np.isfinite(d.llr.values)
    srows.append({"matcher": name, "condition": cond, "method": method,
                  "Cllr": cllr(d.llr.values[ok], y[ok]), "Cllr_min": cllr_min(d.llr.values[ok], y[ok]),
                  "FNMR@FMR1e-3": float((d[d.mated == 1].score < thr).mean()),
                  "FMR max/min": G.FMR.max() / G.FMR.min() if G.FMR.min() > 0 else np.inf,
                  "LR spread": G.LR_shift_rel.max() - G.LR_shift_rel.min(),
                  "Cllr worst/best group": G.Cllr.max() / G.Cllr.min(),
                  "top FMR": G.loc[G.FMR.idxmax(), "group"],
                  **({"FMR women/men": float((nm[nm.gender == "woman"].score >= thr).mean() /
                                             max((nm[nm.gender == "man"].score >= thr).mean(), 1e-12)),
                      "FNMR women/men": float((d[(d.mated == 1) & (d.gender == "woman")].score < thr).mean() /
                                              max((d[(d.mated == 1) & (d.gender == "man")].score < thr).mean(), 1e-12))}
                     if set(d.gender) >= {"woman", "man"} else {}), "largest LR overstatement": G.loc[G.LR_shift_rel.idxmax(), "group"],
                  **({"Cllr_synthcal": cllr(t_llr, y)} if t_llr is not None else {})})


def load_synth_cal(name, cond):
    """Optional: logistic calibration fitted on the synthetic set, saved by tools/save_synth_cal.py."""
    cls = {"original": "edit_hq", "cctv_mid": "cctv_still", "cctv_low": "cctv_still"}[cond]
    f = OUT / f"synth_cal_{name}_{cls}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    s = ScoreLRSystem(transform="none").fit(z["scores"], z["y"])
    return s


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--models", nargs="+", default=["adaface", "adaface_kprpe", "arcface_ir101", "glint_r100"])
    p.add_argument("--per-id", type=int, default=8)
    p.add_argument("--max-ids", type=int, default=0, help="max identities per group (0 = all)")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--cache", default="data/cache_real")
    a = p.parse_args(argv)
    m = manifest(Path(a.root), a.per_id, a.max_ids)
    fc.log(f"{a.name}: {m.identity.nunique()} identities, {len(m)} images, groups {sorted(m.group.unique())}")
    crops, ok = load_crops(m, Path(a.cache) / a.name)
    m = m.reset_index(drop=True)
    P = make_pairs(m, a.k)
    fc.log(f"pairs: {int(P.mated.sum())} mated, {int((P.mated == 0).sum())} non-mated")
    coh = synth_cohort()
    rows, srows = [], []
    for name in a.models:
        order = "bgr"
        f = ROOT / "data/embeddings_v2" / f"{name}_channel_order.json"
        if f.exists():
            order = json.loads(f.read_text())["channel_order"]
        Eo = fc.embed(name, crops, ok, order)
        Ec = fc.embed(name, coh, np.ones(len(coh), bool), order)
        refs = P.ia.values
        for cond, prm in DEGR.items():
            E = Eo if prm is None else fc.embed(name, degrade_all(crops, prm, seed=1), ok, order)
            raw = (Eo[P.ia.values] * E[P.ib.values]).sum(1)
            raw[~(ok[P.ia.values] & ok[P.ib.values])] = np.nan
            C = E @ Ec.T
            mu_a, sd_a = C.mean(1), np.maximum(C.std(1), 1e-6)
            top = -np.sort(-C, axis=1)[:, :100]
            mu_t, sd_t = top.mean(1), np.maximum(top.std(1), 1e-6)
            pb = P.ib.values
            for method, sc in [("raw", raw), ("T-norm (synthetic cohort)", (raw - mu_a[pb]) / sd_a[pb]),
                               ("top-100 T-norm", (raw - mu_t[pb]) / sd_t[pb])]:
                analyse(P, sc, name, cond, method, rows, srows,
                        load_synth_cal(name, cond) if method == "raw" else None)
            fc.log(f"{name}: {cond} done")
        del Eo
    R, S = pd.DataFrame(rows), pd.DataFrame(srows)
    R.to_csv(OUT / f"real_mated_{a.name}.csv", index=False); S.to_csv(OUT / f"real_mated_{a.name}_summary.csv", index=False)
    md = [f"# Real faces with identities ({a.name}): mated + non-mated per group\n",
          f"{m.identity.nunique()} identities, {len(m)} images; reference original, probes at three qualities; "
          "threshold FMR 1e-3; LR logistic + ELUB, identity-disjoint CV; cohort = 600 synthetic references.\n",
          "## Summary\n", S.round(3).to_string(index=False)]
    for name in a.models:
        for col in ["LR_shift_rel", "FMR_ratio", "Cllr"]:
            t = R[R.matcher == name].pivot_table(index="group", columns=["condition", "method"], values=col)
            md += [f"\n## {name}: {col}\n", t.round(2).to_string()]
    (OUT / f"real_mated_{a.name}.md").write_text("\n".join(md), encoding="utf-8")
    fc.log(f"-> results_extended/real_mated_{a.name}.md")


if __name__ == "__main__":
    main()
