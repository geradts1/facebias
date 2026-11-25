"""
Extended study on the synthetic set (runs on the PC with the GPUs, every step resumable).

    python tools/extended_study.py                      # all steps
    python tools/extended_study.py --steps crops embed  # only some steps
    python tools/extended_study.py --cache data/cache

Steps
  crops      detect + align every image ONCE (SCRFD 1x/2x, YuNet fallback), store the 112x112
             crops, plus face-image quality per crop: eDifFIQA (Babnik et al. 2024, OpenCV zoo,
             CC-BY-4.0), sharpness (Laplacian variance) and brightness.
             Images: the final set (system_c_export/images.csv) AND the 1233 identities that were
             removed as near-duplicates (used as look-alikes / doppelgangers).
  embed      embeddings from the crop cache for every matcher in facebias.embed.MATCHERS
  score      score the pair list per matcher -> data/scored_<m>_v2.csv
  analyse    per-condition bias + LR direction per matcher (same code as for AdaFace/System C)
  compare    cross-matcher comparison: is the group pattern the same for every matcher?
  generator  generator check: group homogeneity and FMR per generator (Z-Image vs Qwen),
             near-duplicate removal rate per group
  quality    quality-aware LR calibration vs global / per condition / per group
  doppel     look-alike (doppelganger) non-mated pairs: FMR and LR for near-duplicate identities
Outputs: results_extended/*.md, *.csv, *.png  (+ results_by_condition_<m>_v2, results_lr_direction_<m>_v2)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

EXPORT = ROOT / "data/system_c_export"
SYNTH = ROOT / "data/synth1000"
EMB2 = ROOT / "data/embeddings_v2"
OUT = ROOT / "results_extended"
CHUNK = 2000
CLASS_OF = None  # filled lazily from analyze_by_condition.CLASSES
EDIFFIQA_URL = ("https://huggingface.co/opencv/face_image_quality_assessment_ediffiqa/resolve/main/"
                "ediffiqa_tiny_jun2024.onnx")


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def local_path(p):
    return ROOT / Path(str(p).replace("\\", "/"))


def fname(p):
    return Path(str(p).replace("\\", "/")).name


def generator_of(identity_id):
    s = str(identity_id)
    return "Z-Image" if s[:5] in ("syn26", "syn27", "syn28", "syn29") else "Qwen"


def md_table(df, digits=4):
    try:
        return df.round(digits).to_markdown()
    except ImportError:
        return "```\n" + df.round(digits).to_string() + "\n```"


def condition_class(cond: pd.Series) -> pd.Series:
    from analyze_by_condition import CLASSES
    out = pd.Series("other", index=cond.index, dtype=object)
    c = cond.astype(str)
    for k, sel in CLASSES.items():
        out[sel(c).values] = k
    out[c.eq("reference").values] = "reference"
    return out


# ------------------------------------------------------------------ image index
def image_index(cache: Path) -> pd.DataFrame:
    f = cache / "image_index.csv"
    if f.exists():
        return pd.read_csv(f)
    main = pd.read_csv(EXPORT / "images.csv")
    main = pd.DataFrame({"file": main.file, "path": main.path, "identity_id": main.identity_id,
                         "condition": main.condition, "set": "main"})
    allm = pd.read_csv(SYNTH / "metadata_all.csv", low_memory=False)
    dup = allm[allm.excluded.astype(str).eq("near_duplicate")]
    dup = pd.DataFrame({"file": dup.path.map(fname), "path": dup.path, "identity_id": dup.identity_id,
                        "condition": dup.condition, "set": "dup"})
    idx = pd.concat([main, dup], ignore_index=True).drop_duplicates("file").reset_index(drop=True)
    cache.mkdir(parents=True, exist_ok=True)
    idx.to_csv(f, index=False)
    log(f"image index: {len(main)} final-set + {len(dup)} near-duplicate images -> {f}")
    return idx


# ------------------------------------------------------------------ step: crops (+ quality)
def ediffiqa():
    import cv2
    f = ROOT / "models" / "ediffiqa_tiny_jun2024.onnx"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading eDifFIQA -> {f}")
        urllib.request.urlretrieve(EDIFFIQA_URL, f)
    return cv2.dnn.readNetFromONNX(str(f))


def quality(net, crop_bgr):
    import cv2
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = np.moveaxis(((rgb / 255.0) - 0.5) / 0.5, -1, 0)[None]
    net.setInput(x)
    q = float(np.ravel(net.forward())[0])
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return q, float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())


def step_crops_parallel(cache: Path, idx: pd.DataFrame, workers: int):
    """Detection is the slow part (~16 img/s in one process): run several shards in parallel,
    spread over the GPUs; each shard writes its own chunks, so this is resumable too."""
    import os
    import subprocess
    if workers <= 1:
        return step_crops(cache, idx)
    try:
        import torch
        n_gpu = max(1, torch.cuda.device_count())
    except Exception:
        n_gpu = 1
    procs = []
    for i in range(workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % n_gpu))
        logf = open(ROOT / "logs" / f"extended_crops_shard{i}.log", "a")
        procs.append(subprocess.Popen([sys.executable, "-u", __file__, "--steps", "crops", "--cache", str(cache),
                                       "--shard", f"{i}/{workers}"], env=env, stdout=logf, stderr=subprocess.STDOUT))
    log(f"crops: {workers} shards started over {n_gpu} GPU(s); logs/extended_crops_shard*.log")
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"crops shard failed: exit codes {codes}")


def step_crops(cache: Path, idx: pd.DataFrame, shard=(0, 1)):
    import os
    import cv2
    from insightface.utils import face_align
    from facebias.embed import Aligner
    from fte_second_detector import detect, to_arcface_order, yunet
    cdir = cache / "crops"
    cdir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(idx) + CHUNK - 1) // CHUNK
    todo = [c for c in range(n_chunks) if c % shard[1] == shard[0] and not (cdir / f"chunk_{c:05d}.npz").exists()]
    if not todo:
        log("crops: all chunks present")
        return
    aligner, yn, qnet = Aligner(), yunet(), ediffiqa()
    t0 = time.time()
    for k, c in enumerate(todo):
        part = idx.iloc[c * CHUNK:(c + 1) * CHUNK]
        crops, det, q, blur, bright = [], [], [], [], []
        for p in part.path:
            img = cv2.imread(str(local_path(p)))
            cr, d = None, 0
            if img is not None:
                cr = aligner(img)
                d = 1 if cr is not None else 0
                if cr is None:
                    r = detect(yn, img)
                    if r is not None:
                        cr, d = face_align.norm_crop(img, to_arcface_order(r[2]), image_size=112), 2
            if cr is None:
                crops.append(np.zeros((112, 112, 3), np.uint8)); det.append(0)
                q.append(np.nan); blur.append(np.nan); bright.append(np.nan)
            else:
                crops.append(cr); det.append(d)
                a, b, g = quality(qnet, cr)
                q.append(a); blur.append(b); bright.append(g)
        tmp = cdir / f"chunk_{c:05d}.tmp.npz"
        np.savez(tmp, files=part.file.values, crops=np.array(crops, np.uint8),
                 det=np.array(det, np.int8), q=np.array(q, np.float32), blur=np.array(blur, np.float32),
                 bright=np.array(bright, np.float32))
        os.replace(tmp, cdir / f"chunk_{c:05d}.npz")      # atomic: a half-written chunk is never used
        rate = (k + 1) * CHUNK / (time.time() - t0)
        d = np.array(det)
        log(f"crops chunk {c + 1}/{n_chunks}: SCRFD {np.mean(d == 1):.1%} YuNet {np.mean(d == 2):.1%} "
            f"none {np.mean(d == 0):.1%}  ~{rate:.0f} img/s, eta {(len(todo) - k - 1) * CHUNK / rate / 60:.0f} min")


def iter_crops(cache):
    for f in sorted((cache / "crops").glob("chunk_*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        yield f.stem, np.load(f, allow_pickle=True)


def quality_table(cache) -> pd.DataFrame:
    f = cache / "quality.csv"
    if f.exists():
        return pd.read_csv(f)
    rows = []
    for _, z in iter_crops(cache):
        rows.append(pd.DataFrame({"file": z["files"], "det": z["det"], "q": z["q"],
                                  "blur": z["blur"], "bright": z["bright"]}))
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(f, index=False)
    return t


# ------------------------------------------------------------------ step: embed
def choose_channels(cache, idx, name):
    """d' between reference and ref2 (same person) vs different references, for rgb and bgr."""
    from facebias.embed import make_matcher
    EMB2.mkdir(parents=True, exist_ok=True)
    f = EMB2 / f"{name}_channel_order.json"
    if f.exists():
        return json.loads(f.read_text())["channel_order"]
    lut = {}
    for _, z in iter_crops(cache):
        for fn, cr, d in zip(z["files"], z["crops"], z["det"]):
            if d > 0 and ("__reference." in fn or "__ref2__" in fn):
                lut[fn] = cr
        if len(lut) > 600:
            break
    ids = {}
    for fn in lut:
        ids.setdefault(fn.split("__")[0], {})["ref" if "__reference." in fn else "ref2"] = fn
    pairs = [(v["ref"], v["ref2"]) for v in ids.values() if len(v) == 2][:150]
    res = {}
    for order in ("rgb", "bgr"):
        m = make_matcher(name, channel_order=order)
        ea = m([lut[a] for a, _ in pairs]); eb = m([lut[b] for _, b in pairs])
        mated = (ea * eb).sum(1)
        nonm = (ea @ ea.T)[np.triu_indices(len(ea), 1)]
        res[order] = float((mated.mean() - nonm.mean()) / np.sqrt(0.5 * (mated.var() + nonm.var())))
        del m
    best = max(res, key=res.get)
    f.write_text(json.dumps({"channel_order": best, "d_prime": res, "n": len(pairs)}, indent=1))
    log(f"{name}: channel order {best} (d' {res})")
    return best


def step_embed(cache, idx, models):
    from facebias.embed import MATCHERS, make_matcher
    for name in models:
        out = EMB2 / name
        out.mkdir(parents=True, exist_ok=True)
        chunks = [f.stem for f in sorted((cache / "crops").glob("chunk_*.npz")) if not f.name.endswith(".tmp.npz")]
        if all((out / f"{s}.npz").exists() for s in chunks):
            log(f"{name}: embeddings complete")
            continue
        order = choose_channels(cache, idx, name) if MATCHERS[name][0].startswith("cvl") else "bgr"
        net = make_matcher(name, channel_order=order)
        t0 = time.time()
        for k, (stem, z) in enumerate(iter_crops(cache)):
            if (out / f"{stem}.npz").exists():
                continue
            ok = z["det"] > 0
            e = None
            sel = np.where(ok)[0]
            crops = z["crops"]
            for i in range(0, len(sel), 128):
                b = sel[i:i + 128]
                v = net([crops[j] for j in b])
                if e is None:
                    e = np.zeros((len(ok), v.shape[1]), np.float16)
                e[b] = v.astype(np.float16)
            if e is None:
                e = np.zeros((len(ok), 512), np.float16)
            np.savez(out / f"{stem}.npz", files=z["files"], emb=e, ok=ok)
            if k % 10 == 0:
                log(f"{name}: {stem} ({(k + 1) * CHUNK / (time.time() - t0):.0f} img/s)")
        del net
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def load_emb(name):
    files, embs, oks = [], [], []
    for f in sorted((EMB2 / name).glob("chunk_*.npz")):
        z = np.load(f, allow_pickle=True)
        files += list(z["files"]); embs.append(z["emb"].astype(np.float32)); oks.append(z["ok"])
    E = np.concatenate(embs)
    return {f: i for i, f in enumerate(files)}, E, np.concatenate(oks)


# ------------------------------------------------------------------ step: score
def step_score(models):
    pairs0 = pd.read_csv(EXPORT / "pairs_meta.csv")
    fa0, fb0 = pairs0.image_a.map(fname), pairs0.image_b.map(fname)
    for name in models:
        out = ROOT / f"data/scored_{name}_v2.csv"
        if out.exists():
            continue
        idxm, E, ok = load_emb(name)
        ia, ib = fa0.map(idxm), fb0.map(idxm)
        have = ia.notna() & ib.notna()
        s = np.full(len(pairs0), np.nan)
        a, b = ia[have].astype(int).values, ib[have].astype(int).values
        good = ok[a] & ok[b]
        v = np.full(len(a), np.nan)
        v[good] = (E[a[good]] * E[b[good]]).sum(1)
        s[have.values] = v
        p = pairs0.copy()
        p["score"], p["fte"], p["matcher"] = s, np.isnan(s), name
        p.to_csv(out, index=False)
        log(f"{name}: scored {np.isfinite(s).mean():.1%} of {len(p)} pairs -> {out.name}")


# ------------------------------------------------------------------ step: analyse
def step_analyse(models):
    from analyze_by_condition import analyse
    from lr_direction import direction, report
    for name in models:
        n = f"{name}_v2"
        p = pd.read_csv(ROOT / f"data/scored_{name}_v2.csv")
        try:
            if not (ROOT / f"results_by_condition_{n}.csv").exists():
                analyse(p, n)
            if not (ROOT / f"results_lr_direction_{n}.csv").exists():
                report(direction(p, n, "none"), n)
        except Exception as e:      # keep going with the other matchers / steps
            log(f"analyse {n} FAILED: {type(e).__name__}: {e}")


# ------------------------------------------------------------------ step: compare
def step_compare(models):
    from facebias.embed import MATCHERS
    OUT.mkdir(exist_ok=True)
    res = {m: pd.read_csv(ROOT / f"results_by_condition_{m}_v2.csv") for m in models
           if (ROOT / f"results_by_condition_{m}_v2.csv").exists()}
    lrd = {m: pd.read_csv(ROOT / f"results_lr_direction_{m}_v2.csv") for m in models
           if (ROOT / f"results_lr_direction_{m}_v2.csv").exists()}
    md = ["# Vergelijking van matchers op de synthetische set\n",
          "Zelfde paren, zelfde crops (één detectie+alignering voor alle matchers, YuNet als tweede detector), "
          "zelfde analyse. Per conditieklasse een eigen drempel bij FMR = 1e-3. Vraag: is het groepspatroon "
          "matcher-specifiek of zit het in de data (generator)?\n",
          "| matcher | omschrijving |\n|---|---|"] + [f"| {m} | {MATCHERS[m][2]} |" for m in res] + [""]
    rows = []
    for cls in ["edit_hq", "cctv_still", "video_I", "video_P", "passfoto"]:
        for g in ["ancestry", "gender_presentation", "age_band"]:
            fm = {}
            for m, r in res.items():
                t = r[(r["class"] == cls) & (r.group_by == g)].set_index("group")
                if len(t):
                    fm[m] = t.FMR_bal
            if len(fm) < 2:
                continue
            F = pd.DataFrame(fm)
            # FMR relative to the median group of that matcher (log10), so matchers are comparable
            rel = np.log10(F.replace(0, np.nan)).sub(np.log10(F.replace(0, np.nan)).median())
            corr = F.rank().corr(method="pearson")      # Spearman on group ranks
            off = corr.values[np.triu_indices(len(corr), 1)]
            rows.append({"class": cls, "group_by": g, "mean_spearman": np.nanmean(off),
                         "min_spearman": np.nanmin(off)})
            if g == "ancestry" or cls in ("video_P", "cctv_still"):
                md.append(f"\n## {cls} · {g}: FMR_bal (gebalanceerd), Spearman tussen matchers gemiddeld "
                          f"{np.nanmean(off):.2f} (min {np.nanmin(off):.2f})\n")
                md.append(md_table(F.assign(**{"log10 t.o.v. mediaan (gem.)": rel.mean(axis=1)})
                                   .sort_values("log10 t.o.v. mediaan (gem.)", ascending=False), 5))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "matchers_rank_agreement.csv", index=False)
    md.insert(4 + len(res), "\n## Samenvatting: overeenstemming in groepsvolgorde (Spearman van FMR_bal tussen matchers)\n\n"
              + md_table(summ.set_index(["class", "group_by"]), 2) + "\n")
    # LR direction agreement (relative mated bias per ancestry, video_P)
    if lrd:
        t = pd.DataFrame({m: d[(d["class"] == "video_P") & (d.group_by == "ancestry")].set_index("group")
                          .bias_mated_rel for m, d in lrd.items()})
        if len(t):
            md.append("\n## LR-richting (video_P, mated, relatief): > 0 = globale LR te hoog voor die groep\n")
            md.append(md_table(t.assign(gemiddeld=t.mean(axis=1)).sort_values("gemiddeld", ascending=False), 2))
    _compare_plot(res)
    (OUT / "matchers_comparison.md").write_text("\n".join(md), encoding="utf-8")
    log("compare -> results_extended/matchers_comparison.md")


def _compare_plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = ["edit_hq", "cctv_still", "video_P"]
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 7), sharey=True)
    order = None
    for ax, cls in zip(axes, classes):
        F = pd.DataFrame({m: r[(r["class"] == cls) & (r.group_by == "ancestry")].set_index("group").FMR_bal
                          for m, r in res.items()})
        L = np.log10(F.replace(0, np.nan)); L = L.sub(L.median())
        if order is None:
            order = L.mean(axis=1).sort_values().index.tolist()
        L = L.reindex(order)
        im = ax.imshow(L.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(L.shape[1])); ax.set_xticklabels(L.columns, rotation=45, ha="right", fontsize=8)
        ax.set_title(cls)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6, label="log10 FMR t.o.v. mediane groep")
    fig.suptitle("FMR per herkomst en matcher (rood = hoger dan de mediane groep)")
    fig.savefig(OUT / "matchers_comparison.png", dpi=130, bbox_inches="tight")


# ------------------------------------------------------------------ step: generator
def step_generator(models, idx):
    from facebias.bias import error_rates, threshold_at_fmr
    OUT.mkdir(exist_ok=True)
    meta = pd.read_csv(SYNTH / "metadata_all.csv", low_memory=False,
                       usecols=["identity_id", "condition", "excluded", "ancestry", "gender_presentation", "age_band"])
    ids = meta[meta.condition == "reference"].drop_duplicates("identity_id").copy()
    ids["generator"] = ids.identity_id.map(generator_of)
    ids["status"] = ids.excluded.fillna("kept").astype(str)
    md = ["# Generator-biastoets\n",
          "Komen de groepsverschillen van de matcher, of van de generator? Drie toetsen:\n",
          "1. **Bijna-duplicaten per groep**: tijdens het bouwen zijn personen verwijderd die te veel op een "
          "ander leken (ArcFace > 0,65). Een hoger percentage betekent dat de generator binnen die groep "
          "minder variatie maakt.",
          "2. **Homogeniteit**: gemiddelde cosine-gelijkenis tussen referentiefoto's van *verschillende* personen "
          "binnen dezelfde groep, per generator en per matcher (hoger = gezichten lijken meer op elkaar).",
          "3. **FMR per generator**: als een groep bij beide generatoren een hoge FMR heeft, is het minder "
          "waarschijnlijk een artefact van één generator.\n"]
    t = ids[ids.status.isin(["kept", "near_duplicate"])]
    nd = (t.assign(nd=t.status.eq("near_duplicate"))
          .groupby(["ancestry", "generator"]).nd.agg(["mean", "size"]).unstack("generator"))
    nd.columns = [f"{a}_{b}" for a, b in nd.columns]
    nd = nd.rename(columns=lambda c: c.replace("mean_", "% dup ").replace("size_", "n "))
    for c in [c for c in nd.columns if c.startswith("% dup")]:
        nd[c] *= 100
    allnd = t.assign(nd=t.status.eq("near_duplicate")).groupby("ancestry").nd.mean() * 100
    nd["% dup totaal"] = allnd
    md += ["\n## 1. Bijna-duplicaten verwijderd per herkomst (%)\n", md_table(nd.sort_values("% dup totaal", ascending=False), 1)]
    nd.to_csv(OUT / "generator_near_duplicates.csv")

    kept = ids[ids.status == "kept"].set_index("identity_id")
    homo_rows, fmr_rows = [], []
    for name in models:
        idxm, E, ok = load_emb(name)
        refs = kept.assign(file=[f"SYNTH_{i}__reference.png" for i in kept.index])
        refs = refs[refs.file.map(lambda f: f in idxm and ok[idxm[f]])]
        R = E[[idxm[f] for f in refs.file]]
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        allmean = float((R.sum(0) @ R.sum(0) - len(R)) / (len(R) * (len(R) - 1)))
        for (anc, gen), g in refs.reset_index().groupby(["ancestry", "generator"]):
            if len(g) < 10:
                continue
            X = R[refs.index.get_indexer(g.identity_id)]
            S = X @ X.T
            v = S[np.triu_indices(len(X), 1)]
            homo_rows.append({"matcher": name, "ancestry": anc, "generator": gen, "n": len(X),
                              "mean_cos": v.mean(), "p99_cos": np.quantile(v, 0.99),
                              "mean_cos_minus_all": v.mean() - allmean})
        # FMR per generator per class (same-demographic non-mated pairs, class threshold over all pairs)
        p = pd.read_csv(ROOT / f"data/scored_{name}_v2.csv",
                        usecols=["id_a", "id_b", "mated", "condition_b", "score", "same_demo", "ancestry_a"])
        p = p.dropna(subset=["score"])
        p["cls"] = condition_class(p.condition_b)
        p["sd"] = p.same_demo.astype(str).str.lower().eq("true")
        p["gen_a"], p["gen_b"] = p.id_a.map(generator_of), p.id_b.map(generator_of)
        for cls, d in p.groupby("cls"):
            if cls not in ("edit_hq", "cctv_still", "video_P") or d.mated.nunique() < 2:
                continue
            thr = threshold_at_fmr(d.score, d.mated, 1e-3)
            nm = d[(d.mated == 0) & d.sd & (d.gen_a == d.gen_b)]
            for (anc, gen), g in nm.groupby(["ancestry_a", "gen_a"]):
                if len(g) < 200:
                    continue
                fmr_rows.append({"matcher": name, "class": cls, "ancestry": anc, "generator": gen,
                                 "n_nonmated": len(g), "FMR": float((g.score >= thr).mean())})
    H = pd.DataFrame(homo_rows); H.to_csv(OUT / "generator_homogeneity.csv", index=False)
    F = pd.DataFrame(fmr_rows); F.to_csv(OUT / "generator_fmr.csv", index=False)
    if len(H):
        h = H.pivot_table(index="ancestry", columns=["generator"], values="mean_cos_minus_all", aggfunc="mean")
        h["gem."] = H.groupby("ancestry").mean_cos_minus_all.mean()
        md += ["\n## 2. Homogeniteit (gemiddelde cosine binnen groep min gemiddelde over alle paren; "
               "gemiddeld over matchers)\n", md_table(h.sort_values("gem.", ascending=False), 3)]
        hm = H.pivot_table(index="ancestry", columns="matcher", values="mean_cos_minus_all", aggfunc="mean")
        md += ["\nPer matcher (beide generatoren samen):\n", md_table(hm, 3)]
    if len(F):
        for cls in ["cctv_still", "video_P"]:
            f = F[F["class"] == cls].pivot_table(index="ancestry", columns="generator", values="FMR", aggfunc="mean")
            if len(f) and f.shape[1] == 2:
                c = f.dropna()
                rho = c.rank().corr().iloc[0, 1] if len(c) > 3 else np.nan
                md += [f"\n## 3. FMR per generator, {cls} (gemiddeld over matchers; alleen paren binnen één generator). "
                       f"Spearman Z-Image vs Qwen over groepen: {rho:.2f}\n",
                       md_table(f.assign(ratio=f.max(axis=1) / f.min(axis=1).replace(0, np.nan)).sort_values("Qwen", ascending=False), 5)]
    if len(H) and len(nd):
        x = H.groupby("ancestry").mean_cos_minus_all.mean()
        y = F[F["class"] == "video_P"].groupby("ancestry").FMR.mean() if len(F) else pd.Series(dtype=float)
        j = pd.concat([x.rename("homogeniteit"), nd["% dup totaal"].rename("dup%"), y.rename("FMR_video_P")], axis=1).dropna()
        if len(j) > 3:
            md += ["\n## Samenhang over herkomstgroepen (Spearman)\n", md_table(j.rank().corr(), 2),
                   "\nEen sterke samenhang tussen homogeniteit en FMR wijst erop dat (een deel van) de FMR-verschillen "
                   "door de generator komt: personen binnen die groep zijn minder verschillend gegenereerd."]
    (OUT / "generator_check.md").write_text("\n".join(md), encoding="utf-8")
    log("generator -> results_extended/generator_check.md")


# ------------------------------------------------------------------ step: quality-aware LR
def _cllr(llr, y):
    llr, y = np.asarray(llr, float), np.asarray(y)
    lr = np.power(10.0, np.clip(llr, -15, 15))
    m, n = lr[y == 1], lr[y == 0]
    return 0.5 * (np.mean(np.log2(1 + 1 / m)) + np.mean(np.log2(1 + n)))


def _cv_llr(X, y, groups, n_splits=5, seed=0):
    """Identity-disjoint CV logistic calibration -> out-of-fold log10 LR (posterior log-odds minus
    the training prior log-odds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = np.asarray(X, float).reshape(len(y), -1)
    out = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2:
        return out
    k = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], y[tr])
        prior = np.log(y[tr].mean() / (1 - y[tr].mean()))
        out[te] = (clf.decision_function(X[te]) - prior) / np.log(10)
    return out


def step_quality(models, cache):
    OUT.mkdir(exist_ok=True)
    qt = quality_table(cache).set_index("file")
    md = ["# Kwaliteit-afhankelijke LR-kalibratie\n",
          "Beeldkwaliteit per gezicht: **eDifFIQA** (Babnik et al. 2024; hoger = beter), berekend op dezelfde "
          "112x112 crops. Kalibratievarianten (logistische regressie, 5-voudige CV gegroepeerd op de referentiepersoon, "
          "probe-condities behalve de referentie, alleen paren binnen dezelfde demografische groep):\n",
          "- **M0 globaal**: één model op de score",
          "- **M1 per conditie**: apart model per conditieklasse (vereist dat de conditie bekend is)",
          "- **M2 per kwaliteitsklasse**: apart model per kwintiel eDifFIQA van de probe",
          "- **M3 score+kwaliteit**: één model op score, kwaliteit en hun product\n",
          "Per methode wordt dezelfde methode óók binnen elke groep apart gefit. bias = gemiddelde (log10 LR "
          "methode − log10 LR dezelfde methode per groep). **spreiding** = verschil tussen de groep met de hoogste "
          "en de laagste bias; **abs_rel_bias** = gemiddelde afwijking van een groep t.o.v. het groepsgemiddelde. "
          "Kleiner = de methode benadeelt groepen minder. Cllr = kwaliteit van de LR's (lager = beter).\n"]
    meta_q = None
    for name in models:
        p = pd.read_csv(ROOT / f"data/scored_{name}_v2.csv",
                        usecols=["image_a", "image_b", "id_a", "id_b", "mated", "condition_b", "score",
                                 "same_demo", "ancestry_a", "gender_presentation_a", "age_band_a"])
        p = p.dropna(subset=["score"])
        p = p[p.same_demo.astype(str).str.lower().eq("true")].reset_index(drop=True)
        p["cls"] = condition_class(p.condition_b)
        p = p[p.cls.isin(["passfoto", "edit_hq", "cctv_still", "video_I", "video_P"])].reset_index(drop=True)
        fb = p.image_b.map(fname)
        p["q"] = fb.map(qt.q).values
        p = p.dropna(subset=["q"]).reset_index(drop=True)
        if meta_q is None:
            meta_q = p.drop_duplicates("image_b")[["ancestry_a", "gender_presentation_a", "age_band_a", "cls", "q"]]
        y, s, q, grp = p.mated.values.astype(int), p.score.values, p.q.values, p.id_a.values
        qs = (q - q.mean()) / q.std()
        p["qbin"] = pd.qcut(q, 5, labels=False, duplicates="drop")
        cls_arr, qbin_arr = p.cls.values, p.qbin.values

        def fit(method, ix):
            """Out-of-fold log10 LR of a calibration method, fitted on the pairs ix only."""
            out = np.full(len(ix), np.nan)
            if method == "M0 globaal":
                return _cv_llr(s[ix], y[ix], grp[ix])
            if method == "M3 score+kwaliteit":
                return _cv_llr(np.c_[s[ix], qs[ix], s[ix] * qs[ix]], y[ix], grp[ix])
            key = cls_arr[ix] if method == "M1 per conditie" else qbin_arr[ix]
            for k in pd.unique(key):
                sel = key == k
                out[sel] = _cv_llr(s[ix][sel], y[ix][sel], grp[ix][sel])
            return out

        methods = ["M0 globaal", "M1 per conditie", "M2 per kwaliteit", "M3 score+kwaliteit"]
        allix = np.arange(len(p))
        L = {m: fit(m, allix) for m in methods}
        rows = []
        for g in ["ancestry_a", "gender_presentation_a", "age_band_a"]:
            for c, ix in p.groupby(g).groups.items():
                ix = np.asarray(ix)
                for m in methods:
                    spec = fit(m, ix)                     # the same method, fitted within the group only
                    v = np.isfinite(L[m][ix]) & np.isfinite(spec)
                    yy, d = y[ix][v], (L[m][ix] - spec)[v]
                    rows.append({"matcher": name, "group_by": g.replace("_a", ""), "group": c, "model": m,
                                 "Cllr": _cllr(L[m][ix][v], yy), "Cllr_groupspecific": _cllr(spec[v], yy),
                                 "bias_mated": d[yy == 1].mean(), "bias_nonmated": d[yy == 0].mean()})
        R = pd.DataFrame(rows)
        for c in ["bias_mated", "bias_nonmated"]:
            R[c + "_rel"] = R[c] - R.groupby(["group_by", "model"])[c].transform("mean")
        R.to_csv(OUT / f"quality_lr_{name}.csv", index=False)
        md.append(f"\n## {name}\n")
        for g in ["ancestry", "gender_presentation", "age_band"]:
            r = R[R.group_by == g]
            s1 = r.groupby("model").agg(
                Cllr_gem=("Cllr", "mean"), Cllr_groepsspecifiek_gem=("Cllr_groupspecific", "mean"),
                spreiding_bias_mated=("bias_mated", lambda x: x.max() - x.min()),
                abs_rel_bias_mated=("bias_mated_rel", lambda x: np.abs(x).mean()),
                abs_rel_bias_nonmated=("bias_nonmated_rel", lambda x: np.abs(x).mean()))
            md += [f"\n### {g}: samenvatting over groepen\n", md_table(s1.reindex(methods), 3)]
            if g == "ancestry":
                piv = r.pivot_table(index="group", columns="model", values="bias_mated_rel")[methods]
                md += ["\nrelatieve bias_mated per herkomst (log10; > 0 = globale LR te hoog voor die groep):\n",
                       md_table(piv.sort_values("M0 globaal", ascending=False), 2)]
        log(f"quality LR done for {name}")
    if meta_q is not None:
        qa = meta_q.pivot_table(index="ancestry_a", columns="cls", values="q", aggfunc="mean")
        md += ["\n## Is de kwaliteitsmaat zelf groepsafhankelijk? Gemiddelde eDifFIQA per herkomst en conditie\n",
               md_table(qa.sort_values("video_P") if "video_P" in qa else qa, 3),
               "\nVerschillen hier betekenen dat kwaliteitsgebaseerde kalibratie ook groepsverschillen kan "
               "overnemen of juist corrigeren; beoordeel dat samen met de tabellen hierboven."]
    (OUT / "quality_lr.md").write_text("\n".join(md), encoding="utf-8")
    log("quality -> results_extended/quality_lr.md")


# ------------------------------------------------------------------ step: doppelgangers
def step_doppel(models, idx):
    from facebias.bias import threshold_at_fmr
    OUT.mkdir(exist_ok=True)
    meta = pd.read_csv(SYNTH / "metadata_all.csv", low_memory=False,
                       usecols=["identity_id", "condition", "excluded", "ancestry", "gender_presentation", "age_band"])
    demo = meta.drop_duplicates("identity_id").set_index("identity_id")
    dup_ids = sorted(idx[idx.set == "dup"].identity_id.unique())
    kept_ids = sorted(meta[(meta.condition == "reference") & meta.excluded.isna()].identity_id.unique())
    # pair every removed near-duplicate with its most similar kept identity, judged by AdaFace (not used in building)
    idxm, E, ok = load_emb("adaface")
    def refvec(i):
        f = f"SYNTH_{i}__reference.png"
        return E[idxm[f]] if f in idxm and ok[idxm[f]] else None
    K = [(i, refvec(i)) for i in kept_ids]; K = [(i, v) for i, v in K if v is not None]
    KM = np.stack([v for _, v in K]); KM /= np.linalg.norm(KM, axis=1, keepdims=True)
    pairs = []
    for b in dup_ids:
        v = refvec(b)
        if v is None:
            continue
        sim = KM @ (v / np.linalg.norm(v))
        j = int(np.argmax(sim))
        pairs.append((K[j][0], b, float(sim[j])))
    P = pd.DataFrame(pairs, columns=["id_a", "id_b", "adaface_ref_cos"])
    probes = idx[idx.set == "dup"][["file", "identity_id", "condition"]].rename(columns={"identity_id": "id_b"})
    H = P.merge(probes, on="id_b")
    H["file_a"] = "SYNTH_" + H.id_a + "__reference.png"
    H["cls"] = condition_class(H.condition)
    for c in ["ancestry", "gender_presentation", "age_band"]:
        H[f"{c}_a"] = H.id_a.map(demo[c]); H[f"{c}_b"] = H.id_b.map(demo[c])
    H["same_ancestry"] = H.ancestry_a == H.ancestry_b
    md = ["# Dubbelgangers (look-alikes) als lastige non-mated paren\n",
          f"De {len(P)} personen die bij het bouwen als bijna-duplicaat zijn verwijderd, zijn gekoppeld aan de "
          "meest gelijkende persoon in de definitieve set (op de referentiefoto, volgens AdaFace). Non-mated paren: "
          "referentiefoto van de behouden persoon tegen álle beelden van de dubbelganger. Drempel per conditieklasse "
          "en LR-kalibratie komen uit de gewone set (zelfde matcher), dus dit is wat er gebeurt als zo'n "
          "dubbelganger in een zaak opduikt.\n",
          "Let op: de bijna-duplicaten zijn geselecteerd met ArcFace (cosine > 0,65); voor ArcFace-achtige matchers "
          "is deze set dus per constructie lastig.\n",
          f"Zelfde herkomst binnen het paar: {P.merge(H[['id_b','same_ancestry']].drop_duplicates(), on='id_b').same_ancestry.mean():.0%}.\n"]
    rows = []
    for name in models:
        idxm, E, ok = load_emb(name)
        ia, ib = H.file_a.map(idxm), H.file.map(idxm)
        have = ia.notna() & ib.notna()
        sc = np.full(len(H), np.nan)
        a, b = ia[have].astype(int).values, ib[have].astype(int).values
        g = ok[a] & ok[b]
        v = np.full(len(a), np.nan); v[g] = (E[a[g]] * E[b[g]]).sum(1)
        sc[have.values] = v
        H[f"score_{name}"] = sc
        base = pd.read_csv(ROOT / f"data/scored_{name}_v2.csv",
                           usecols=["mated", "condition_b", "score", "same_demo", "id_a"]).dropna(subset=["score"])
        base["cls"] = condition_class(base.condition_b)
        base = base[base.same_demo.astype(str).str.lower().eq("true")]
        from sklearn.linear_model import LogisticRegression
        for cls in ["passfoto", "edit_hq", "cctv_still", "video_I", "video_P"]:
            d = base[base.cls == cls]
            hh = H[(H.cls == cls) & np.isfinite(H[f"score_{name}"])]
            if d.empty or hh.empty or d.mated.nunique() < 2:
                continue
            thr = threshold_at_fmr(d.score, d.mated, 1e-3)
            clf = LogisticRegression(C=1.0, max_iter=500).fit(d[["score"]].values, d.mated.values)
            prior = np.log(d.mated.mean() / (1 - d.mated.mean()))
            llr = (clf.decision_function(hh[[f"score_{name}"]].values) - prior) / np.log(10)
            nm = d[d.mated == 0]
            llr_n = (clf.decision_function(nm[["score"]].values) - prior) / np.log(10)
            for anc, ix in [("alle", np.arange(len(hh)))] + [(k, np.where(hh.ancestry_a.values == k)[0])
                                                              for k in sorted(hh.ancestry_a.dropna().unique())]:
                if len(ix) < 30:
                    continue
                rows.append({"matcher": name, "class": cls, "ancestry": anc, "n": len(ix),
                             "FMR_doppel": float((hh[f"score_{name}"].values[ix] >= thr).mean()),
                             "FMR_gewoon": float((nm.score >= thr).mean()) if anc == "alle" else np.nan,
                             "LR>1_doppel": float((llr[ix] > 0).mean()),
                             "LR>100_doppel": float((llr[ix] > 2).mean()),
                             "LR>1_gewoon": float((llr_n > 0).mean()) if anc == "alle" else np.nan,
                             "median_log10LR_doppel": float(np.median(llr[ix]))})
    R = pd.DataFrame(rows); R.to_csv(OUT / "doppelgangers.csv", index=False)
    H.to_csv(OUT / "doppelganger_pairs.csv", index=False)
    if len(R):
        a = R[R.ancestry == "alle"].set_index(["class", "matcher"]).drop(columns=["ancestry"])
        md += ["\n## Alle dubbelgangers\n", md_table(a, 4)]
        for cls in ["cctv_still", "video_P", "edit_hq"]:
            r = R[(R["class"] == cls) & (R.ancestry != "alle")]
            if len(r):
                piv = r.pivot_table(index="ancestry", columns="matcher", values="LR>1_doppel")
                md += [f"\n## {cls}: aandeel dubbelganger-paren met LR > 1 per herkomst\n",
                       md_table(piv.assign(gem=piv.mean(axis=1)).sort_values("gem", ascending=False), 3)]
    (OUT / "doppelgangers.md").write_text("\n".join(md), encoding="utf-8")
    log("doppel -> results_extended/doppelgangers.md")


# ------------------------------------------------------------------ main
STEPS = ["crops", "embed", "score", "analyse", "compare", "generator", "quality", "doppel"]


def main(argv=None):
    from facebias.embed import MATCHERS
    p = argparse.ArgumentParser()
    p.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    p.add_argument("--models", nargs="+", default=list(MATCHERS), choices=list(MATCHERS))
    p.add_argument("--cache", default=str(ROOT / "data/cache_extended"))
    p.add_argument("--workers", type=int, default=4, help="parallel detection processes for the crops step")
    p.add_argument("--shard", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    cache = Path(a.cache)
    idx = image_index(cache)
    if a.shard:                                   # child process of the parallel crops step
        i, n = map(int, a.shard.split("/"))
        step_crops(cache, idx, (i, n))
        return
    log(f"steps {a.steps}; matchers {a.models}; {len(idx)} images; cache {cache}")
    for s in a.steps:
        t0 = time.time()
        if s == "crops":
            step_crops_parallel(cache, idx, a.workers)
        elif s == "embed":
            step_embed(cache, idx, a.models)
        elif s == "score":
            step_score(a.models)
        elif s == "analyse":
            step_analyse(a.models)
        elif s == "compare":
            step_compare(a.models)
        elif s == "generator":
            step_generator(a.models, idx)
        elif s == "quality":
            step_quality(a.models, cache)
        elif s == "doppel":
            step_doppel(a.models, idx)
        log(f"step {s} done in {(time.time() - t0) / 60:.1f} min")
    log("ALL DONE")


if __name__ == "__main__":
    main()
