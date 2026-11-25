"""
Score the System C pair list with open-source recognisers (AdaFace, ArcFace)
and run the same bias / LR analysis as for System C.

    python tools/run_open_matchers.py                       # adaface + arcface, everything
    python tools/run_open_matchers.py --models adaface      # only AdaFace
    python tools/run_open_matchers.py --steps embed score   # skip evaluation

Steps (each resumable):
  embed    detect + align every image once, embed with each model
           -> data/embeddings/<model>/chunk_XXXXX.npz   (+ fte list)
  score    cosine similarity for every pair in data/system_c_export/pairs_meta.csv
           -> data/scored_<model>.csv   (same columns as the System C import)
  evaluate bias + LR analysis per group (transform none: cosine scores)
           -> results_<model>/<group>/report.html, plus results_open_matchers.md (side by side)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPORT = ROOT / "data/system_c_export"
EMB = ROOT / "data/embeddings"
GROUPS = ["ancestry", "gender_presentation", "age_band"]
CHUNK = 2000


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def local_path(p):
    return ROOT / Path(str(p).replace("\\", "/"))


# ---------------------------------------------------------------- embed
def choose_adaface_channels(images, aligner):
    """AdaFace's original release expects BGR input, the HF/CVLface wrapper RGB.
    Decide empirically on reference vs ref2 (same person) and reference vs
    other references (different persons): keep the order with the larger d'."""
    import cv2
    from facebias.embed import AdaFaceEmbedder
    f = EMB / "adaface_channel_order.json"
    if f.exists():
        return json.loads(f.read_text())["channel_order"]
    refs = images[images.condition == "reference"].head(150)
    ref2 = images[images.condition == "ref2"].set_index("identity_id")
    crops_a, crops_b = [], []
    for _, r in refs.iterrows():
        if r.identity_id not in ref2.index:
            continue
        a = aligner(cv2.imread(str(local_path(r.path))))
        b = aligner(cv2.imread(str(local_path(ref2.loc[r.identity_id, "path"]))))
        if a is not None and b is not None:
            crops_a.append(a); crops_b.append(b)
    res = {}
    for order in ("rgb", "bgr"):
        m = AdaFaceEmbedder(channel_order=order)
        ea, eb = m(crops_a), m(crops_b)
        mated = (ea * eb).sum(1)
        S = ea @ ea.T
        nonm = S[np.triu_indices(len(ea), 1)]
        res[order] = float((mated.mean() - nonm.mean()) / np.sqrt(0.5 * (mated.var() + nonm.var())))
        del m
    best = max(res, key=res.get)
    f.write_text(json.dumps({"channel_order": best, "d_prime": res, "n": len(crops_a)}, indent=1))
    log(f"AdaFace channel order: {best}  (d' {res})")
    return best


def embed(models, images):
    import cv2
    from facebias.embed import Aligner, EMBEDDERS
    EMB.mkdir(parents=True, exist_ok=True)
    aligner = Aligner()
    kw = {"adaface": {"channel_order": choose_adaface_channels(images, aligner)}} if "adaface" in models else {}
    nets = {m: EMBEDDERS[m](**kw.get(m, {})) for m in models}
    for m in models:
        (EMB / m).mkdir(exist_ok=True)
    n_chunks = (len(images) + CHUNK - 1) // CHUNK
    t0 = time.time()
    for c in range(n_chunks):
        todo = [m for m in models if not (EMB / m / f"chunk_{c:05d}.npz").exists()]
        if not todo:
            continue
        part = images.iloc[c * CHUNK:(c + 1) * CHUNK]
        crops, ok = [], []
        for p in part.path:
            img = cv2.imread(str(local_path(p)))
            cr = aligner(img) if img is not None else None
            ok.append(cr is not None)
            crops.append(cr if cr is not None else np.zeros((112, 112, 3), np.uint8))
        ok = np.array(ok)
        for m in todo:
            embs = []
            for i in range(0, len(crops), 64):
                embs.append(nets[m](crops[i:i + 64]))
            e = np.concatenate(embs).astype(np.float16)
            np.savez(EMB / m / f"chunk_{c:05d}.npz", files=part.file.values, emb=e, ok=ok)
        rate = (c + 1) * CHUNK / (time.time() - t0)
        log(f"embed chunk {c + 1}/{n_chunks}  FTE {100 * (~ok).mean():.1f}%  ~{rate:.0f} img/s")


def load_embeddings(model):
    files, embs, oks = [], [], []
    for f in sorted((EMB / model).glob("chunk_*.npz")):
        z = np.load(f, allow_pickle=True)
        files += list(z["files"]); embs.append(z["emb"].astype(np.float32)); oks.append(z["ok"])
    return {f: i for i, f in enumerate(files)}, np.concatenate(embs), np.concatenate(oks)


# ---------------------------------------------------------------- score
def score(model):
    idx, E, ok = load_embeddings(model)
    pairs = pd.read_csv(EXPORT / "pairs_meta.csv")
    fa = pairs.image_a.map(lambda p: Path(str(p).replace("\\", "/")).name).map(idx)
    fb = pairs.image_b.map(lambda p: Path(str(p).replace("\\", "/")).name).map(idx)
    have = fa.notna() & fb.notna()
    ia, ib = fa[have].astype(int).values, fb[have].astype(int).values
    s = np.full(len(pairs), np.nan)
    s[have.values] = (E[ia] * E[ib]).sum(1)
    fte = np.zeros(len(pairs), bool)
    fte[have.values] = ~(ok[ia] & ok[ib])
    s[fte] = np.nan
    pairs["score"], pairs["fte"], pairs["matcher"] = s, fte, model
    out = ROOT / f"data/scored_{model}.csv"
    pairs.to_csv(out, index=False)
    log(f"{model}: {len(pairs)} pairs scored -> {out}; FTE pairs {100 * fte.mean():.2f}%, "
        f"missing {int((~have).sum())}")
    for g in ["ancestry_b", "gender_presentation_b", "age_band_b", "condition_b"]:
        t = pairs.groupby(g).fte.mean().mul(100).round(2)
        print(f"  FTE% per {g}: " + ", ".join(f"{k}={v}" for k, v in t.items()))


# ---------------------------------------------------------------- evaluate
def evaluate(model, n_boot):
    from facebias.cli import _evaluate
    pairs = pd.read_csv(ROOT / f"data/scored_{model}.csv")
    for g in GROUPS:
        out = ROOT / f"results_{model}/{g}"
        if (out / "summary.json").exists():
            continue
        log(f"evaluate {model} by {g}")
        _evaluate(pairs, g, out, "logit", "none", 1e-3, n_boot, 5)


def side_by_side(models):
    lines = ["# Open-source referentiesystemen op de synthetische set\n",
             "Cosine-scores, globale drempel bij FMR = 1e-3 (over alle paren), LR via logistische regressie "
             "+ ELUB, identiteit-disjuncte kruisvalidatie. ArcFace (buffalo_l) is ook gebruikt bij het "
             "bouwen van de set en is daardoor optimistisch; AdaFace niet.\n"]
    for g in GROUPS:
        lines.append(f"\n## {g}\n")
        tabs = {}
        for m in models:
            d = ROOT / f"results_{m}/{g}"
            if not (d / "score_level_bias.csv").exists():
                continue
            s = pd.read_csv(d / "score_level_bias.csv", index_col=0)
            l = pd.read_csv(d / "lr_level_bias.csv", index_col=0)
            t = pd.DataFrame({f"{m} FMR": s.FMR, f"{m} FNMR": s.FNMR, f"{m} d'": s.d_prime,
                              f"{m} Cllr glob": l.get("global_Cllr"),
                              f"{m} Cllr_cal glob": l.get("global_Cllr_cal"),
                              f"{m} Cllr_cal spec": l.get("specific_Cllr_cal")})
            tabs[m] = t
            summ = json.loads((d / "summary.json").read_text())
            fmt = lambda v: "∞ (laagste groep 0)" if v > 1e6 else f"{v:.1f}"
            lines.append(f"- **{m}**: FMR max/min {fmt(summ.get('FMR_max/min', float('nan')))}, "
                         f"FNMR max/min {fmt(summ.get('FNMR_max/min', float('nan')))}, "
                         f"FDR {summ.get('FDR(alpha=0.5)', float('nan')):.3f}")
        if tabs:
            tab = pd.concat(tabs.values(), axis=1).round(4)
            try:
                lines.append("\n" + tab.to_markdown() + "\n")
            except ImportError:
                lines.append("\n```\n" + tab.to_string() + "\n```\n")
    out = ROOT / "results_open_matchers.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"side-by-side summary -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["adaface", "arcface"], choices=["adaface", "arcface"])
    p.add_argument("--steps", nargs="+", default=["embed", "score", "evaluate"])
    p.add_argument("--n-boot", type=int, default=30)
    a = p.parse_args(argv)
    images = pd.read_csv(EXPORT / "images.csv")
    log(f"{len(images)} images, models {a.models}, steps {a.steps}")
    if "embed" in a.steps:
        embed(a.models, images)
    for m in a.models:
        if "score" in a.steps:
            score(m)
        if "evaluate" in a.steps:
            evaluate(m, a.n_boot)
    if "evaluate" in a.steps:
        side_by_side(a.models)
    log("done")


if __name__ == "__main__":
    main()
