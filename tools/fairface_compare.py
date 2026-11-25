"""
Generator check against REAL faces: does a matcher see the faces of a group as more alike in the
synthetic set than in a real, demographically balanced set (FairFace)?

FairFace (Karkkainen & Joo, WACV 2021; CC-BY-4.0): face crops from public Flickr images with
race / gender / age labels. Only the validation split, only adults (age class 20+), only used
locally for this measurement; nothing from FairFace is stored in the synthetic set or published.

Measure per group, per matcher (different persons only):
  homogeneity  = mean cosine between faces of different persons in the group, minus the mean over
                 all pairs of that source (so real and synthetic are on their own baseline)
  tail ratio   = share of pairs above the source's 99.9th percentile (all pairs), relative to the
                 source's overall share (= 0.001): a proxy for how much more often false matches
                 occur within that group
Synthetic side: per person one reference photo and one situation edit (hq), comparable to the
in-the-wild variation of FairFace. Synthetic groups are mapped to FairFace's 7 race labels.

    python tools/fairface_compare.py [--models adaface arcface ...] [--max-per-group 1200]
Output: results_extended/fairface_comparison.md / .csv / .png
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "results_extended"
SYNTH = ROOT / "data/synth1000"

FF_RACE = {0: "East Asian", 1: "Indian", 2: "Black", 3: "White", 4: "Middle Eastern",
           5: "Latino_Hispanic", 6: "Southeast Asian"}
FF_AGE = {3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"}
SYN_TO_FF = {
    "Northern European": "White", "Southern European": "White", "Eastern European": "White",
    "West African": "Black", "East African": "Black", "Afro-Caribbean / Afro-Surinamese": "Black",
    "East Asian": "East Asian",
    "Southeast Asian": "Southeast Asian", "Indonesian": "Southeast Asian",
    "South Asian": "Indian", "Hindustani-Surinamese": "Indian",
    "Middle Eastern": "Middle Eastern", "North African": "Middle Eastern", "Turkish": "Middle Eastern",
    "Latin American (mestizo)": "Latino_Hispanic",
    # not mapped: Indigenous American, Pacific Islander, Mixed ancestry
}
LIMIT = 0  # test only
BOOT = 1000
MIN_N = 50
FF_FILE = "0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet"
FF_FILES_FULL = [FF_FILE, "0.25/train-00000-of-00002-d405faba4f4b9b85.parquet",
                 "0.25/train-00001-of-00002-dd3cb68164727418.parquet"]
SUFFIX = ""


def log(m):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True)


def md_table(df, d=3):
    try:
        return df.round(d).to_markdown()
    except ImportError:
        return "```\n" + df.round(d).to_string() + "\n```"


def align_all(images_bgr):
    from insightface.utils import face_align
    from facebias.embed import Aligner
    from fte_second_detector import detect, to_arcface_order, yunet
    al, yn = Aligner(), yunet()
    crops, ok = [], []
    for k, img in enumerate(images_bgr):
        cr = al(img) if img is not None else None
        if cr is None and img is not None:
            r = detect(yn, img)
            if r is not None:
                cr = face_align.norm_crop(img, to_arcface_order(r[2]), image_size=112)
        ok.append(cr is not None)
        crops.append(cr if cr is not None else np.zeros((112, 112, 3), np.uint8))
        if (k + 1) % 2000 == 0:
            log(f"  aligned {k + 1}/{len(images_bgr)}")
    return np.array(crops), np.array(ok)


def load_fairface(cache):
    import cv2
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = cache / "fairface_crops.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return pd.DataFrame({"race": z["race"], "gender": z["gender"], "age": z["age"]}), z["crops"], z["ok"]
    p = hf_hub_download("HuggingFaceM4/FairFace", FF_FILE, repo_type="dataset", local_dir=str(cache / "fairface"))
    t = pq.read_table(p).to_pandas()
    t = t[t.age >= 3].reset_index(drop=True)                     # adults only
    if LIMIT:
        t = t.groupby("race").head(LIMIT).reset_index(drop=True)
    imgs = [cv2.imdecode(np.frombuffer(b["bytes"], np.uint8), cv2.IMREAD_COLOR) for b in t.image]
    log(f"FairFace: {len(t)} adult faces; aligning")
    crops, ok = align_all(imgs)
    meta = pd.DataFrame({"race": t.race.map(FF_RACE), "gender": t.gender.map({0: "man", 1: "woman"}),
                         "age": t.age.map(FF_AGE)})
    np.savez(f, crops=crops, ok=ok, race=meta.race.values, gender=meta.gender.values, age=meta.age.values)
    return meta, crops, ok


def load_fairface_full(cache):
    """Validation + training split (all adults), aligned per parquet file (resumable)."""
    import cv2
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    metas, crops, oks = [], [], []
    for k, fn in enumerate(FF_FILES_FULL):
        f = cache / f"fairface_crops_part{k}.npz"
        if not f.exists():
            p = hf_hub_download("HuggingFaceM4/FairFace", fn, repo_type="dataset", local_dir=str(cache / "fairface"))
            t = pq.read_table(p).to_pandas()
            t = t[t.age >= 3].reset_index(drop=True)
            log(f"FairFace part {k}: {len(t)} adult faces; aligning")
            imgs = [cv2.imdecode(np.frombuffer(b["bytes"], np.uint8), cv2.IMREAD_COLOR) for b in t.image]
            c, o = align_all(imgs)
            del imgs
            np.savez(f, crops=c, ok=o, race=t.race.map(FF_RACE).values,
                     gender=t.gender.map({0: "man", 1: "woman"}).values, age=t.age.map(FF_AGE).values)
        z = np.load(f, allow_pickle=True)
        metas.append(pd.DataFrame({"race": z["race"], "gender": z["gender"], "age": z["age"]}))
        crops.append(z["crops"]); oks.append(z["ok"])
    return pd.concat(metas, ignore_index=True), np.concatenate(crops), np.concatenate(oks)


def load_synth_from_cache(cache):
    """All reference photos + all hq situation edits of the final set, from the crop cache of
    tools/extended_study.py (same detection/alignment as everything else)."""
    idx = pd.read_csv(cache / "image_index.csv")
    want = idx[(idx.set == "main") & idx.condition.isin(["reference", "hq"])]
    keep = set(want.file)
    m = pd.read_csv(SYNTH / "metadata.csv", low_memory=False).drop_duplicates("identity_id").set_index("identity_id")
    rows, crops = [], []
    for f in sorted((cache / "crops").glob("chunk_*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f, allow_pickle=True)
        for fn, cr, d in zip(z["files"], z["crops"], z["det"]):
            if fn in keep and d > 0:
                iid = fn[len("SYNTH_"):].split("__")[0]
                rows.append((iid, "reference" if "__reference." in fn else "hq")); crops.append(cr)
    meta = pd.DataFrame(rows, columns=["identity_id", "kind"])
    for c, src in [("ancestry", "ancestry"), ("gender", "gender_presentation"), ("age", "age_band")]:
        meta[c] = meta.identity_id.map(m[src])
    log(f"synthetic from crop cache: {len(meta)} faces of {meta.identity_id.nunique()} persons")
    return meta, np.array(crops), np.ones(len(meta), bool)


def load_synth(cache, seed=0):
    import cv2
    f = cache / "synth_refhq_crops.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return pd.DataFrame({k: z[k] for k in ["identity_id", "ancestry", "gender", "age", "kind"]}), z["crops"], z["ok"]
    m = pd.read_csv(SYNTH / "metadata.csv", low_memory=False)
    rng = np.random.default_rng(seed)
    refs = m[m.condition == "reference"]
    hq = m[m.condition == "hq"].groupby("identity_id").apply(
        lambda g: g.iloc[rng.integers(len(g))], include_groups=False).reset_index()
    sel = pd.concat([refs.assign(kind="reference"), hq.assign(kind="hq")], ignore_index=True)
    log(f"synthetic: {len(sel)} images (reference + one hq edit per person); aligning")
    imgs = [cv2.imread(str(ROOT / Path(str(p).replace("\\", "/")))) for p in sel.path]
    crops, ok = align_all(imgs)
    meta = pd.DataFrame({"identity_id": sel.identity_id.values, "ancestry": sel.ancestry.values,
                         "gender": sel.gender_presentation.values, "age": sel.age_band.values,
                         "kind": sel.kind.values})
    np.savez(f, crops=crops, ok=ok, **{k: meta[k].values for k in meta})
    return meta, crops, ok


def embed(name, crops, ok, order):
    from facebias.embed import make_matcher
    net = make_matcher(name, channel_order=order)
    E = None
    idx = np.where(ok)[0]
    for i in range(0, len(idx), 128):
        b = idx[i:i + 128]
        v = net([crops[j] for j in b])
        if E is None:
            E = np.zeros((len(crops), v.shape[1]), np.float32)
        E[b] = v
    del net
    return E


def group_stats(E, groups, person=None, max_n=1200, seed=0):
    """Homogeneity and tail ratio per group, pairs of different persons only."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups, dtype=object)                 # Arrow-backed pandas arrays do not support [:, None]
    person = None if person is None else np.asarray(person, dtype=object)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    # baseline over all pairs (random sample)
    n = len(E)
    a, b = rng.integers(n, size=400_000), rng.integers(n, size=400_000)
    keep = a != b if person is None else person[a] != person[b]
    allc = (E[a[keep]] * E[b[keep]]).sum(1)
    base, thr = allc.mean(), np.quantile(allc, 0.999)
    rows = []
    for g in pd.unique(groups):
        if pd.isna(g):
            continue
        ix = np.where(groups == g)[0]
        if len(ix) < MIN_N:
            continue
        if len(ix) > max_n:
            ix = rng.choice(ix, max_n, replace=False)
        S = E[ix] @ E[ix].T
        valid = ~np.eye(len(ix), dtype=bool)
        if person is not None:
            valid &= person[ix][:, None] != person[ix][None, :]
        cnt = valid.sum(1).astype(float)
        rm = np.where(valid, S, 0).sum(1) / np.maximum(cnt, 1)            # per face: mean cos to other persons
        rt = np.where(valid, S > thr, False).sum(1) / np.maximum(cnt, 1)  # per face: share above the tail threshold
        bs = rng.integers(len(ix), size=(BOOT, len(ix)))                  # bootstrap over faces
        hb, tb = rm[bs].mean(1) - base, rt[bs].mean(1) / 0.001
        rows.append({"group": g, "n": len(ix), "homogeneity": rm.mean() - base,
                     "homogeneity_lo": np.quantile(hb, 0.025), "homogeneity_hi": np.quantile(hb, 0.975),
                     "tail_ratio": rt.mean() / 0.001,
                     "tail_ratio_lo": np.quantile(tb, 0.025), "tail_ratio_hi": np.quantile(tb, 0.975)})
    return pd.DataFrame(rows, columns=["group", "n", "homogeneity", "homogeneity_lo", "homogeneity_hi",
                                       "tail_ratio", "tail_ratio_lo", "tail_ratio_hi"]).set_index("group")


def main(argv=None):
    from facebias.embed import MATCHERS
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MATCHERS), choices=list(MATCHERS))
    p.add_argument("--cache", default=str(ROOT / "data/cache_extended"))
    p.add_argument("--max-per-group", type=int, default=1200)
    p.add_argument("--limit", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--full", action="store_true",
                   help="FairFace train+validation and all synthetic reference+hq faces from the crop cache")
    a = p.parse_args(argv)
    global LIMIT, MIN_N
    LIMIT = a.limit
    if a.limit:
        MIN_N = 5
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True); OUT.mkdir(exist_ok=True)
    global SUFFIX
    if a.full:
        SUFFIX = "_full"
        ffm, ffc, ffok = load_fairface_full(cache)
        sm, sc, sok = load_synth_from_cache(cache)
    else:
        ffm, ffc, ffok = load_fairface(cache)
        sm, sc, sok = load_synth(cache)
    log(f"FairFace faces found {ffok.mean():.1%}; synthetic {sok.mean():.1%}")
    ffm, ffc = ffm[ffok].reset_index(drop=True), ffc[ffok]
    sm, sc = sm[sok].reset_index(drop=True), sc[sok]
    sm["ff_race"] = sm.ancestry.map(SYN_TO_FF)
    rows = []
    import json
    for name in a.models:
        order = "bgr"
        f = ROOT / "data/embeddings_v2" / f"{name}_channel_order.json"
        if f.exists():
            order = json.loads(f.read_text())["channel_order"]
        Ef, Es = embed(name, ffc, np.ones(len(ffc), bool), order), embed(name, sc, np.ones(len(sc), bool), order)
        person = sm.identity_id.values
        for dim, gf, gs in [("race", ffm.race.values, sm.ff_race.values),
                            ("gender", ffm.gender.values, sm.gender.values),
                            ("age", ffm.age.values, None)]:
            r = group_stats(Ef, gf, max_n=a.max_per_group).add_prefix("real_")
            if gs is not None:
                s = group_stats(Es, gs, person, max_n=a.max_per_group).add_prefix("synth_")
                r = r.join(s, how="outer")
            rows.append(r.assign(matcher=name, dim=dim).reset_index())
        s2 = group_stats(Es, sm.ancestry.values, person, max_n=a.max_per_group)
        rows.append(s2.add_prefix("synth_").assign(matcher=name, dim="ancestry_synth").reset_index())
        log(f"{name}: done")
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(OUT / f"fairface_comparison{SUFFIX}.csv", index=False)
    report(R)


def report(R):
    for c in [c for c in R.columns if c.startswith(("real_", "synth_"))]:
        R[c] = pd.to_numeric(R[c], errors="coerce")
    md = ["# Synthetische gezichten vs echte gezichten (FairFace)\n",
          "Vraag: ziet een matcher de gezichten binnen een groep in de synthetische set als méér op elkaar lijkend "
          "dan bij echte gezichten van die groep? Zo ja, dan komt (een deel van) de hogere FMR voor die groep door de "
          "generator en niet door de matcher.\n",
          "FairFace: validatieset, alleen volwassenen, CC-BY-4.0, alleen lokaal gebruikt voor deze meting. "
          "Synthetisch: per persoon de referentiefoto + één situatie-edit. Alleen paren van verschillende personen.\n",
          "- **homogeniteit**: gemiddelde cosine binnen de groep min het gemiddelde over alle paren van dezelfde bron.",
          "- **tail-ratio**: hoe vaak een paar binnen de groep boven het 99,9e percentiel van alle paren (van die bron) "
          "uitkomt, gedeeld door 0,1%. Een benadering van 'hoeveel vaker valse matches binnen deze groep'.\n"]
    race = R[R.dim == "race"]
    if "synth_homogeneity" in race:
        h = race.pivot_table(index="group", columns="matcher", values=["real_homogeneity", "synth_homogeneity"])
        diff = (race.assign(d=race.synth_homogeneity - race.real_homogeneity)
                .pivot_table(index="group", columns="matcher", values="d"))
        diff["gem."] = diff.mean(axis=1)
        md += ["\n## Herkomst: synthetisch min echt (homogeniteit; > 0 = synthetische groep homogener dan echt)\n",
               md_table(diff.sort_values("gem.", ascending=False), 3)]
        tr = race.assign(r=np.log2((race.synth_tail_ratio / race.real_tail_ratio).replace([np.inf, 0], np.nan))).pivot_table(
            index="group", columns="matcher", values="r")
        tr["gem."] = tr.mean(axis=1)
        md += ["\n## Herkomst: tail-ratio synthetisch t.o.v. echt (log2; +1 = twee keer zo vaak hoge scores binnen de groep)\n",
               md_table(tr.sort_values("gem.", ascending=False), 2)]
        cors = []
        for m, g in race.groupby("matcher"):
            g = g.dropna(subset=["real_homogeneity", "synth_homogeneity"])
            cors.append({"matcher": m, "Spearman homogeniteit echt vs synth":
                         g.real_homogeneity.rank().corr(g.synth_homogeneity.rank()),
                         "Spearman tail-ratio echt vs synth": g.real_tail_ratio.rank().corr(g.synth_tail_ratio.rank())})
        md += ["\n## Komt de volgorde van de groepen overeen?\n", md_table(pd.DataFrame(cors).set_index("matcher"), 2),
               "\nHoge correlatie: de synthetische set geeft dezelfde groepsvolgorde als echte gezichten (de matcher "
               "'ziet' dezelfde verschillen). Lage correlatie en grote positieve verschillen voor specifieke groepen: "
               "generator-artefact voor die groepen.\n"]
        md += ["\n## Ruwe waarden herkomst\n", md_table(h, 3)]
    for dim in ["gender", "age"]:
        d = R[R.dim == dim]
        cols = [c for c in ["real_homogeneity", "synth_homogeneity", "real_tail_ratio", "synth_tail_ratio"] if c in d]
        md += [f"\n## {dim}\n", md_table(d.pivot_table(index="group", columns="matcher", values=cols), 3)]
    s = R[R.dim == "ancestry_synth"]
    if len(s):
        md += ["\n## Synthetisch per herkomstgroep (alle 18)\n",
               md_table(s.pivot_table(index="group", columns="matcher", values="synth_homogeneity"), 3)]
    (OUT / f"fairface_comparison{SUFFIX}.md").write_text("\n".join(md), encoding="utf-8")
    _plot(race)
    log(f"-> results_extended/fairface_comparison{SUFFIX}.md")


def _plot(race):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "synth_homogeneity" not in race:
        return
    ms = list(race.matcher.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 4.2), squeeze=False)
    for ax, m in zip(axes[0], ms):
        g = race[race.matcher == m].dropna(subset=["real_homogeneity", "synth_homogeneity"])
        ax.scatter(g.real_homogeneity, g.synth_homogeneity)
        for _, r in g.iterrows():
            ax.annotate(r.group, (r.real_homogeneity, r.synth_homogeneity), fontsize=7)
        lo = min(g.real_homogeneity.min(), g.synth_homogeneity.min()); hi = max(g.real_homogeneity.max(), g.synth_homogeneity.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("echt (FairFace)"); ax.set_ylabel("synthetisch"); ax.set_title(m)
    fig.suptitle("Homogeniteit per herkomstgroep: boven de lijn = synthetische groep homogener dan echt")
    fig.tight_layout()
    fig.savefig(OUT / f"fairface_comparison{SUFFIX}.png", dpi=130)


if __name__ == "__main__":
    main()
