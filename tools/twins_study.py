"""
Attribute twins as hard non-mated pairs.

For a stratified sample of persons from the final set, facebias.synth.build (config with
`twins_of`) generated an 'attribute twin': a new person with exactly the same attributes
(ancestry, age, gender, face shape, nose, eyes, hair, headwear, ...) and the same situations,
but a different seed. This is the realistic worst case for a forensic comparison: an innocent
person who matches the description on every recorded attribute.

Pairs (per matcher, per condition class):
  twin non-mated : original reference vs every twin image, and twin reference vs every original probe
  twin mated     : twin reference vs twin probes (sanity check: twins behave like normal persons)
Threshold (FMR = 1e-3) and LR calibration per class come from the normal same-demographic pairs of
the same matcher (data/scored_<m>_v2.csv), so the numbers show what happens when such a person turns up.

    python tools/twins_study.py --twins data/twins
Output: results_extended/twins.md / .csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import extended_study as es  # noqa: E402

OUT = ROOT / "results_extended"
EMB_T = ROOT / "data/embeddings_twins"
CLS = ["passfoto", "edit_hq", "cctv_still", "video_I", "video_P"]


def parse_condition(fn):
    parts = Path(fn).stem.split("__")
    if parts[1] == "reference":
        return "reference"
    if parts[1] == "ref2":
        return "ref2"
    if len(parts) == 3:
        return parts[2]
    return f"{parts[2]}_{parts[3]}"


def twin_index(twins: Path, cache: Path) -> pd.DataFrame:
    f = cache / "image_index.csv"
    if f.exists():
        return pd.read_csv(f)
    rows = []
    for jf in sorted((twins / "identities").glob("*/identity.json")):
        rec = json.loads(jf.read_text())
        if rec.get("status") != "done":
            continue
        for p in sorted(jf.parent.glob("SYNTH_*.png")):
            rows.append({"file": p.name, "path": str(p), "identity_id": rec["identity_id"],
                         "twin_of": rec["attrs"].get("twin_of"), "condition": parse_condition(p.name), "set": "twin"})
    idx = pd.DataFrame(rows)
    cache.mkdir(parents=True, exist_ok=True)
    idx.to_csv(f, index=False)
    es.log(f"twins: {idx.identity_id.nunique()} persons, {len(idx)} images")
    return idx


def calibrators(name):
    """Per class: threshold at FMR 1e-3 and a logistic score->LR calibration from normal pairs."""
    from sklearn.linear_model import LogisticRegression
    from facebias.bias import threshold_at_fmr
    b = pd.read_csv(ROOT / f"data/scored_{name}_v2.csv",
                    usecols=["id_a", "mated", "condition_b", "score", "same_demo", "ancestry_a",
                             "gender_presentation_a", "age_band_a"]).dropna(subset=["score"])
    b["cls"] = es.condition_class(b.condition_b)
    b = b[b.same_demo.astype(str).str.lower().eq("true")]
    out = {}
    for cls in CLS:
        d = b[b.cls == cls]
        if d.mated.nunique() < 2:
            continue
        thr = threshold_at_fmr(d.score, d.mated, 1e-3)
        clf = LogisticRegression(C=1.0, max_iter=500).fit(d[["score"]].values, d.mated.values)
        prior = np.log(d.mated.mean() / (1 - d.mated.mean()))
        nm = d[d.mated == 0]
        out[cls] = {"thr": thr, "clf": clf, "prior": prior,
                    "normal_fmr": nm.assign(f=nm.score >= thr).groupby("ancestry_a").f.mean(),
                    "normal_fmr_all": float((nm.score >= thr).mean())}
    return out


def llr(c, s):
    return (c["clf"].decision_function(np.asarray(s).reshape(-1, 1)) - c["prior"]) / np.log(10)


def boot_mean(x, cluster, B=2000, seed=0):
    """Cluster bootstrap of the mean of x (clusters = original person)."""
    x, cluster = np.asarray(x, float), np.asarray(cluster)
    u, inv = np.unique(cluster, return_inverse=True)
    s, n = np.bincount(inv, weights=x), np.bincount(inv).astype(float)
    w = np.random.default_rng(seed).multinomial(len(u), np.full(len(u), 1 / len(u)), size=B)
    with np.errstate(invalid="ignore"):
        b = (w @ s) / (w @ n)
    return np.nanquantile(b, [0.025, 0.975])


def main(argv=None):
    from facebias.embed import MATCHERS
    p = argparse.ArgumentParser()
    p.add_argument("--twins", required=True)
    p.add_argument("--cache", default="data/cache_twins")
    p.add_argument("--main-cache", default="data/cache_extended")
    p.add_argument("--models", nargs="+", default=list(MATCHERS))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    cache, twins = Path(a.cache), Path(a.twins)
    idx = twin_index(twins, cache)
    es.step_crops_parallel(cache, idx, a.workers)
    # embed twins with every matcher, reusing the channel order chosen on the main set
    EMB_T.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "data/embeddings_v2").glob("*_channel_order.json"):
        if not (EMB_T / f.name).exists():
            shutil.copy(f, EMB_T / f.name)
    es.EMB2 = EMB_T
    es.step_embed(cache, idx, a.models)

    meta = pd.read_csv(es.SYNTH / "metadata.csv", low_memory=False).drop_duplicates("identity_id").set_index("identity_id")
    main_idx = pd.read_csv(Path(a.main_cache) / "image_index.csv")
    main_idx = main_idx[main_idx.set == "main"]
    tw = idx.copy()
    pairs_rows = []
    for t_id, g in tw.groupby("identity_id"):
        o_id = g.twin_of.iloc[0]
        t_ref = f"SYNTH_{t_id}__reference.png"
        o_ref = f"SYNTH_{o_id}__reference.png"
        for r in g.itertuples():
            if r.condition != "reference":
                pairs_rows.append(("twin_nonmated", o_id, o_ref, "main", r.file, "twin", r.condition))
                pairs_rows.append(("twin_mated", o_id, t_ref, "twin", r.file, "twin", r.condition))
        for r in main_idx[main_idx.identity_id == o_id].itertuples():
            if r.condition != "reference":
                pairs_rows.append(("twin_nonmated", o_id, t_ref, "twin", r.file, "main", r.condition))
    P = pd.DataFrame(pairs_rows, columns=["kind", "orig_id", "file_a", "src_a", "file_b", "src_b", "condition"])
    P["cls"] = es.condition_class(P.condition)
    for c in ["ancestry", "gender_presentation", "age_band"]:
        P[c] = P.orig_id.map(meta[c])
    rows, refsim = [], []
    for name in a.models:
        es.EMB2 = ROOT / "data/embeddings_v2"; mi, ME, mok = es.load_emb(name)
        es.EMB2 = EMB_T; ti, TE, tok = es.load_emb(name)

        def vec(f, src):
            i, E, ok = (mi, ME, mok) if src == "main" else (ti, TE, tok)
            j = i.get(f)
            return E[j] if j is not None and ok[j] else None
        sc = np.full(len(P), np.nan)
        for k, r in enumerate(P.itertuples()):
            va, vb = vec(r.file_a, r.src_a), vec(r.file_b, r.src_b)
            if va is not None and vb is not None:
                sc[k] = float(va @ vb)
        P[f"score_{name}"] = sc
        # reference-to-reference similarity: twin vs original, compared with random same-demo persons
        for t_id, g in tw.groupby("identity_id"):
            o_id = g.twin_of.iloc[0]
            va, vb = vec(f"SYNTH_{o_id}__reference.png", "main"), vec(f"SYNTH_{t_id}__reference.png", "twin")
            if va is not None and vb is not None:
                refsim.append({"matcher": name, "orig_id": o_id, "ancestry": meta.ancestry.get(o_id), "cos": float(va @ vb)})
        cal = calibrators(name)
        for cls, c in cal.items():
            d = P[(P.cls == cls) & np.isfinite(P[f"score_{name}"])]
            for kind in ["twin_nonmated", "twin_mated"]:
                dk = d[d.kind == kind]
                if dk.empty:
                    continue
                s = dk[f"score_{name}"].values
                l = llr(c, s)
                hit = (s >= c["thr"]).astype(float) if kind == "twin_nonmated" else (s < c["thr"]).astype(float)
                for gname, gsel in [("alle", np.ones(len(dk), bool))] + [
                        (v, (dk.ancestry == v).values) for v in sorted(dk.ancestry.dropna().unique())]:
                    if gsel.sum() < 30:
                        continue
                    lo, hi = boot_mean(hit[gsel], dk.orig_id.values[gsel])
                    lo2, hi2 = boot_mean((l[gsel] > 0) if kind == "twin_nonmated" else (l[gsel] < 0), dk.orig_id.values[gsel])
                    nf = c["normal_fmr_all"] if gname == "alle" else c["normal_fmr"].get(gname, np.nan)
                    rows.append({"matcher": name, "class": cls, "kind": kind, "ancestry": gname, "n_pairs": int(gsel.sum()),
                                 "n_persons": dk.orig_id[gsel].nunique(),
                                 "rate": hit[gsel].mean(), "rate_lo": lo, "rate_hi": hi,
                                 "normal_FMR": nf if kind == "twin_nonmated" else np.nan,
                                 "LR_misleading": float(((l[gsel] > 0) if kind == "twin_nonmated" else (l[gsel] < 0)).mean()),
                                 "LR_misleading_lo": lo2, "LR_misleading_hi": hi2,
                                 "LR>100": float((l[gsel] > 2).mean()) if kind == "twin_nonmated" else np.nan,
                                 "median_log10LR": float(np.median(l[gsel]))})
        es.log(f"twins: {name} done")
    R = pd.DataFrame(rows); R.to_csv(OUT / "twins.csv", index=False)
    S = pd.DataFrame(refsim); S.to_csv(OUT / "twins_refsim.csv", index=False)
    P.to_csv(OUT / "twins_pairs.csv", index=False)
    report(R, S, tw)


def report(R, S, tw):
    md = ["# Attribuut-tweelingen: lastige non-mated paren\n",
          f"{tw.identity_id.nunique()} nieuwe personen met precies dezelfde kenmerken (herkomst, leeftijd, gender, "
          "gezichtsvorm, neus, ogen, haar, hoofdbedekking, ...) en dezelfde situaties als een persoon uit de set, maar "
          "een andere persoon (andere seed). Dit is het realistische slechtste geval: een onschuldige die op alle "
          "vastgelegde kenmerken overeenkomt.\n",
          "Drempel (FMR = 0,1%) en LR-kalibratie per conditieklasse komen uit de gewone paren van dezelfde matcher. "
          "`rate` = FMR (non-mated) of FNMR (mated, controle); `LR_misleading` = aandeel LR > 1 (non-mated) of "
          "LR < 1 (mated). 95%-intervallen: bootstrap over personen.\n"]
    if len(S):
        s = S.pivot_table(index="ancestry", columns="matcher", values="cos", aggfunc="mean")
        md += ["\n## Hoe sterk lijkt de tweeling op het origineel? (cosine referentie–referentie, gemiddeld)\n",
               md_table(s.assign(gem=s.mean(axis=1)).sort_values("gem", ascending=False), 3)]
    a = R[R.ancestry == "alle"].copy()
    a["FMR-verhouding t.o.v. gewoon"] = a.rate / a.normal_FMR
    md += ["\n## Alle tweelingen\n", md_table(a.set_index(["class", "kind", "matcher"]).drop(columns=["ancestry"]), 4)]
    for cls in ["edit_hq", "cctv_still", "video_P"]:
        r = R[(R["class"] == cls) & (R.kind == "twin_nonmated") & (R.ancestry != "alle")].copy()
        if r.empty:
            continue
        r["txt"] = [f"{100 * x.rate:.2f} [{100 * x.rate_lo:.2f}; {100 * x.rate_hi:.2f}]" for x in r.itertuples()]
        piv = r.pivot_table(index="ancestry", columns="matcher", values="txt", aggfunc="first")
        order = r.groupby("ancestry").rate.mean().sort_values(ascending=False).index
        md += [f"\n## {cls}: FMR (%) van tweeling-paren per herkomst [95%-BI]\n", piv.reindex(order).to_markdown()]
        q = r.pivot_table(index="ancestry", columns="matcher", values="LR_misleading")
        md += [f"\n{cls}: aandeel tweeling-paren met LR > 1\n", md_table(q.reindex(order), 3)]
    (OUT / "twins.md").write_text("\n".join(md), encoding="utf-8")
    es.log("-> results_extended/twins.md")


def md_table(df, d=3):
    return es.md_table(df, d)


if __name__ == "__main__":
    main()
