"""
Two follow-up analyses on the existing scores (CPU only, no GPU needed).

1. Cohort score normalisation without group labels (Z-, T-, S-norm and top-K T-norm).
   Every score is compared with how the same images score against a fixed, demographically mixed cohort
   of other people. Question: does this shrink the LR shift between groups, and at what accuracy cost?
   Open-source matchers only (needs embeddings; for System C the cohort scores sit in its full matrix on
   the System C workstation).
2. Confounders: are the group differences explained by image properties of the probe (eDifFIQA quality,
   sharpness, brightness, face width) or by skin tone rather than ancestry?
   Open-source matchers and System C (imported from the Drive score file when needed).

    python tools/mitigation_study.py [--steps norm confound] [--cohort 600] [--topk 100]
Output: results_extended/mitigation_norm.md/.csv, results_extended/confounders.md/.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "results_extended"
OPEN = ["adaface", "arcface", "adaface_vit", "arcface_ir101", "adaface_ms1mv2"]
CLS_USED = ["edit_hq", "cctv_still", "video_P"]
FV_DRIVE = Path(os.environ.get("SYSTEM_C_DIR", "data/system_c_run"))


def log(m):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True)


def md_table(df, d=3):
    try:
        return df.round(d).to_markdown()
    except ImportError:
        return "```\n" + df.round(d).to_string() + "\n```"


def fname(p):
    return PureWindowsPath(str(p)).name


def iid_of(fn):
    return fn[len("SYNTH_"):].split("__")[0]


# ------------------------------------------------------------------ 1. cohort normalisation
def cohort_stats(E, ok, files, cohort_idx, topk):
    """Per image: mean/sd of its cosine with the cohort (all, and top-K), own identity excluded."""
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    C = En[cohort_idx]
    c_ids = np.array([iid_of(files[i]) for i in cohort_idx])
    n = len(E)
    mu, sd, mu_k, sd_k = (np.full(n, np.nan) for _ in range(4))
    ids = np.array([iid_of(f) for f in files])
    for s in range(0, n, 20000):
        S = En[s:s + 20000] @ C.T                                   # chunk x cohort
        own = ids[s:s + 20000, None] == c_ids[None, :]
        S = np.where(own, np.nan, S)
        mu[s:s + 20000] = np.nanmean(S, 1); sd[s:s + 20000] = np.nanstd(S, 1)
        Ss = -np.sort(-np.nan_to_num(S, nan=-9), axis=1)[:, :topk]
        mu_k[s:s + 20000] = Ss.mean(1); sd_k[s:s + 20000] = Ss.std(1)
    bad = ~ok
    for a in (mu, sd, mu_k, sd_k):
        a[bad] = np.nan
    return mu, np.maximum(sd, 1e-6), mu_k, np.maximum(sd_k, 1e-6)


def step_norm(cohort_n, topk, seed=0, models=None, tag=""):
    import extended_study as es
    from lr_direction import direction
    from facebias.bias import threshold_at_fmr
    from analyze_by_condition import CLASSES as SEL
    rows, dir_rows = [], []
    for m in (models or OPEN):
        src = ROOT / f"data/scored_{m}_v2.csv"
        if not src.exists():
            continue
        idx, E, ok = es.load_emb(m)
        files = np.array(list(idx.keys()))
        # cohort: references of people, mixed over all groups (no group information used)
        refs = [i for f, i in idx.items() if "__reference." in f and ok[i]]
        rng = np.random.default_rng(seed)
        cohort = np.array(sorted(rng.choice(refs, min(cohort_n, len(refs)), replace=False)))
        mu, sd, mu_k, sd_k = cohort_stats(E, ok, files, cohort, topk)
        p = pd.read_csv(src, usecols=["image_a", "image_b", "id_a", "id_b", "mated", "condition_b", "score",
                                      "same_demo", "ancestry_a", "gender_presentation_a", "age_band_a"])
        p = p.dropna(subset=["score"]).reset_index(drop=True)
        ia = p.image_a.map(fname).map(idx).values.astype(int)
        ib = p.image_b.map(fname).map(idx).values.astype(int)
        s = p.score.values
        variants = {
            "raw": s,
            "Z-norm (reference side)": (s - mu[ia]) / sd[ia],
            "T-norm (probe side)": (s - mu[ib]) / sd[ib],
            "S-norm (both)": 0.5 * ((s - mu[ia]) / sd[ia] + (s - mu[ib]) / sd[ib]),
            f"top-{topk} T-norm": (s - mu_k[ib]) / sd_k[ib],
        }
        for vname, v in variants.items():
            q = p.assign(score=v).dropna(subset=["score"])
            q = q[np.isfinite(q.score)]
            # accuracy at the class threshold
            for cls in CLS_USED:
                d = q[SEL[cls](q.condition_b.astype(str))]
                thr = threshold_at_fmr(d.score, d.mated, 1e-3)
                fnmr = float((d[d.mated == 1].score < thr).mean())
                sd_ = d[d.same_demo.astype(str).str.lower().eq("true")]
                nm = sd_[sd_.mated == 0]
                fmr_g = nm.assign(f=nm.score >= thr).groupby("ancestry_a").f.mean()
                fm = fmr_g.replace(0, np.nan)
                rows.append({"matcher": m, "method": vname, "class": cls, "FNMR@FMR0.1%": fnmr,
                             "FMR max/min (ancestry)": fm.max() / fm.min(),
                             "FMR highest group": fmr_g.idxmax(), "FMR lowest": fmr_g.min()})
            cb = q.condition_b.astype(str)
            keep = np.zeros(len(q), bool)
            for k in CLS_USED:
                keep |= SEL[k](cb).values
            r = direction(q[keep], f"{m}|{vname}", "none")
            if len(r):
                r["bias_mated_rel"] = r.bias_mated - r.groupby(["class", "group_by"]).bias_mated.transform("mean")
                r["bias_nonmated_rel"] = r.bias_nonmated - r.groupby(["class", "group_by"]).bias_nonmated.transform("mean")
                r["matcher"], r["method"] = m, vname
                dir_rows.append(r)
            log(f"{m}: {vname} done")
    A = pd.DataFrame(rows); D = pd.concat(dir_rows, ignore_index=True) if dir_rows else pd.DataFrame()
    A.to_csv(OUT / f"mitigation_norm{tag}_accuracy.csv", index=False); D.to_csv(OUT / f"mitigation_norm{tag}_lr.csv", index=False)
    report_norm(A, D, cohort_n, topk, tag)


def report_norm(A, D, cohort_n, topk, tag=""):
    md = ["# Cohort score normalisation: does it reduce the group bias?\n",
          f"Each score is normalised with the scores of the same images against a fixed cohort of {cohort_n} reference "
          "photos of other people, drawn at random from all groups (no group labels used; own identity excluded). "
          "Z-norm = reference side, T-norm = probe side, S-norm = average of both, top-K = only the K most similar "
          f"cohort faces (K = {topk}). LR shift = mean log10 LR of one global calibration minus a calibration per group "
          "(mated pairs, relative to the group mean); spread = highest minus lowest group. Lower spread = fairer; "
          "FNMR at FMR 0.1% shows the accuracy cost.\n"]
    if len(D):
        s = (D[D.group_by == "ancestry"].groupby(["class", "method", "matcher"]).bias_mated_rel
             .agg(lambda x: x.max() - x.min()).unstack("matcher"))
        s["mean"] = s.mean(axis=1)
        md += ["\n## Spread of the LR shift between ancestry groups (log10, mated pairs)\n", md_table(s, 2)]
        g = (D[D.group_by == "gender_presentation"].groupby(["class", "method"]).bias_mated_rel
             .agg(lambda x: x.max() - x.min()).unstack("method"))
        md += ["\n## Same for gender (mean over matchers)\n", md_table(g, 2)]
    a = A.pivot_table(index=["class", "method"], values=["FNMR@FMR0.1%", "FMR max/min (ancestry)"], aggfunc="mean")
    md += ["\n## Accuracy and FMR ratio (mean over matchers)\n", md_table(a, 3)]
    (OUT / f"mitigation_norm{tag}.md").write_text("\n".join(md), encoding="utf-8")
    log("-> results_extended/mitigation_norm.md")


# ------------------------------------------------------------------ 2. confounders
def ensure_system_c():
    f = ROOT / "data/scored_system_c_v2.csv"
    if f.exists():
        return f
    sc, fte = FV_DRIVE / "system_c_scores.csv.gz", FV_DRIVE / "system_c_fte.txt"
    if not sc.exists():
        log("System C score file not found on Drive; skipped"); return None
    subprocess.run([sys.executable, str(ROOT / "tools/import_external_scores.py"), "--scores", str(sc),
                    "--fte", str(fte), "--version", "DBScan ID 5.9.0 b16-s1-0", "--out", str(f)], check=False)
    return f if f.exists() else None


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def step_confound(cache, boot=200):
    from facebias.bias import threshold_at_fmr
    from analyze_by_condition import CLASSES as SEL
    qt = pd.read_csv(Path(cache) / "quality.csv").set_index("file")
    meta = pd.read_csv(ROOT / "data/synth1000/metadata.csv", low_memory=False,
                       usecols=["path", "face_width_px", "monk_skin_tone", "identity_id"])
    meta["file"] = meta.path.map(fname)
    fw = meta.set_index("file").face_width_px
    skin = meta.drop_duplicates("identity_id").set_index("identity_id").monk_skin_tone
    models = OPEN + (["system_c"] if ensure_system_c() else [])
    rows, coef_rows = [], []
    rng = np.random.default_rng(0)
    for m in models:
        src = ROOT / f"data/scored_{m}_v2.csv"
        if not src.exists():
            continue
        p = pd.read_csv(src, usecols=["image_b", "id_a", "mated", "condition_b", "score", "same_demo", "ancestry_a"])
        p = p.dropna(subset=["score"])
        p = p[p.same_demo.astype(str).str.lower().eq("true")].reset_index(drop=True)
        if m == "system_c":
            p["score"] = np.log(np.clip(p.score, 1e-6, 1 - 1e-6) / (1 - np.clip(p.score, 1e-6, 1 - 1e-6)))
        fb = p.image_b.map(fname)
        p["q"], p["blur"], p["bright"] = fb.map(qt.q).values, np.log1p(fb.map(qt.blur).values), fb.map(qt.bright).values
        p["fw"] = np.log(fb.map(fw).astype(float).values)
        p["skin"] = p.id_a.map(skin).astype(float).values
        for cls in ["cctv_still", "video_P"]:
            d = p[SEL[cls](p.condition_b.astype(str))].dropna(subset=["q", "blur", "bright", "fw", "skin"])
            thr = threshold_at_fmr(d.score, d.mated, 1e-3)
            nm = d[d.mated == 0].copy()
            nm["fm"] = (nm.score >= thr).astype(float)
            # (a) direct standardisation over quality x face-width strata
            nm["stratum"] = pd.qcut(nm.q, 5, labels=False, duplicates="drop").astype(str) + "_" + \
                pd.qcut(nm.fw, 3, labels=False, duplicates="drop").astype(str)
            w = nm.stratum.value_counts(normalize=True)
            raw = nm.groupby("ancestry_a").fm.mean()
            std = nm.groupby(["ancestry_a", "stratum"]).fm.mean().unstack().reindex(columns=w.index)
            adj = (std.fillna(std.mean()) * w).sum(axis=1)
            for g in raw.index:
                rows.append({"matcher": m, "class": cls, "ancestry": g, "FMR_raw": raw[g], "FMR_standardised": adj[g],
                             "mean_quality": nm[nm.ancestry_a == g].q.mean(),
                             "mean_face_width_px": float(np.exp(nm[nm.ancestry_a == g].fw).mean())})
            # (b) regression of the non-mated score (standardised) on ancestry, with and without covariates
            z = (nm.score - nm.score.mean()) / nm.score.std()
            G = pd.get_dummies(nm.ancestry_a).drop(columns=["Northern European"], errors="ignore").astype(float)
            cov = nm[["q", "blur", "bright", "fw"]].apply(lambda c: (c - c.mean()) / c.std())
            skin_d = pd.get_dummies(nm.skin.round().astype(int), prefix="skin").iloc[:, 1:].astype(float)
            ones = np.ones((len(nm), 1))
            X0 = np.hstack([ones, G.values]); X1 = np.hstack([ones, G.values, cov.values])
            X2 = np.hstack([ones, G.values, cov.values, skin_d.values])
            b0, b1, b2 = ols(z.values, X0), ols(z.values, X1), ols(z.values, X2)
            r2 = lambda X, b: 1 - np.var(z.values - X @ b) / np.var(z.values)
            Xs = np.hstack([ones, cov.values, skin_d.values]); bs = ols(z.values, Xs)
            # cluster bootstrap (reference persons) for the adjusted ancestry effects
            ids = nm.id_a.values; u = np.unique(ids); pos = {k: np.where(ids == k)[0] for k in u}
            B = []
            for _ in range(boot):
                sel = np.concatenate([pos[k] for k in rng.choice(u, len(u))])
                B.append(ols(z.values[sel], X2[sel])[1:1 + G.shape[1]])
            B = np.array(B)
            for j, g in enumerate(G.columns):
                coef_rows.append({"matcher": m, "class": cls, "ancestry": g, "effect_raw": b0[1 + j],
                                  "effect_adj_image": b1[1 + j], "effect_adj_image_skin": b2[1 + j],
                                  "adj_lo": np.quantile(B[:, j], .025), "adj_hi": np.quantile(B[:, j], .975)})
            coef_rows.append({"matcher": m, "class": cls, "ancestry": "_R2", "effect_raw": r2(X0, b0),
                              "effect_adj_image": r2(X1, b1), "effect_adj_image_skin": r2(X2, b2),
                              "adj_lo": r2(Xs, bs), "adj_hi": np.nan})
        log(f"confounders: {m} done")
    R, C = pd.DataFrame(rows), pd.DataFrame(coef_rows)
    R.to_csv(OUT / "confounders_fmr.csv", index=False); C.to_csv(OUT / "confounders_regression.csv", index=False)
    report_conf(R, C)


def report_conf(R, C):
    md = ["# Are the group differences explained by image properties or skin tone?\n",
          "Non-mated pairs within the same group, threshold per class at FMR 0.1%. (a) FMR per ancestry raw and after "
          "direct standardisation over 15 strata of probe quality (eDifFIQA quintiles) x face width (terciles). "
          "(b) Linear regression of the standardised non-mated score on ancestry (reference: Northern European), "
          "without and with image covariates (quality, sharpness, brightness, face width) and with Monk skin tone "
          "added. Effects in standard deviations of the non-mated score; 95% intervals from a bootstrap over "
          "reference persons. If the effects stay after adjustment, the differences are not explained by these "
          "properties.\n"]
    for cls in ["cctv_still", "video_P"]:
        r = R[R["class"] == cls]
        t = r.groupby("ancestry")[["FMR_raw", "FMR_standardised", "mean_quality", "mean_face_width_px"]].mean()
        t[["FMR_raw", "FMR_standardised"]] *= 100
        md += [f"\n## {cls}: FMR (%) per ancestry, mean over matchers\n",
               md_table(t.sort_values("FMR_raw", ascending=False), 3)]
        c = C[(C["class"] == cls) & (C.ancestry != "_R2")]
        e = c.groupby("ancestry")[["effect_raw", "effect_adj_image", "effect_adj_image_skin", "adj_lo", "adj_hi"]].mean()
        md += [f"\n### {cls}: ancestry effect on the non-mated score vs Northern European (mean over matchers)\n",
               md_table(e.sort_values("effect_raw", ascending=False), 3)]
        r2 = C[(C["class"] == cls) & (C.ancestry == "_R2")].set_index("matcher")[
            ["effect_raw", "effect_adj_image", "effect_adj_image_skin", "adj_lo"]]
        r2.columns = ["R2 ancestry", "R2 ancestry+image", "R2 ancestry+image+skin", "R2 image+skin (no ancestry)"]
        md += [f"\n{cls}: variance explained (R2)\n", md_table(r2, 4)]
    (OUT / "confounders.md").write_text("\n".join(md), encoding="utf-8")
    log("-> results_extended/confounders.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", nargs="+", default=["confound", "norm"])
    p.add_argument("--cohort", type=int, default=600)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--cache", default="data/cache_extended")
    p.add_argument("--models", nargs="+", default=None, help="open matchers for the norm step (default: the five original ones)")
    p.add_argument("--tag", default="", help="suffix for the norm output files, e.g. _new")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if "confound" in a.steps:
        step_confound(a.cache)
    if "norm" in a.steps:
        step_norm(a.cohort, a.topk, models=a.models, tag=a.tag)
    log("ALL DONE")


if __name__ == "__main__":
    main()
