"""
Top-100 T-norm (and plain T-norm) for System C, computed from the full score matrix on the System C workstation.

For every probe we take its scores against a fixed, mixed cohort of 600 references (every k-th reference of
the gallery, no group information, the probe's own identity excluded), transform them to logit space and
normalise every pair score of that probe:
    T-norm        (logit(s) - mean(cohort)) / sd(cohort)
    top-100 T-norm (logit(s) - mean(top 100 cohort)) / sd(top 100 cohort)

Needs Python 3.7+ with numpy (pip install numpy). Reads the matrix with np.memmap (no 4 GB in memory).

    python external_tnorm.py --matrix <score_matrix_dir> ^
        --pairs pairs_rel.csv.gz --out external_tnorm_scores.csv.gz
Output: CSV (gzip) with image_a,image_b,score_raw,score_tnorm,score_top100 (same relative paths as pairs_rel).
"""
import argparse
import csv
import gzip
import io
import os
import sys
import time

import numpy as np

N_COHORT, TOPK, EPS = 600, 100, 1e-6


def log(m):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), m, flush=True)


def norm_path(p):
    return p.strip().strip('"').replace("/", "\\").lower()


def read_list(path):
    """probes.tsv / gallery.tsv: one row per matrix row/column; take the first field that looks like an image path."""
    out = []
    with io.open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            img = next((x for x in parts if x.lower().endswith((".png", ".jpg", ".jpeg"))), None)
            if img is None:
                if i == 0:
                    continue          # header line
                raise SystemExit("no image path in line %d of %s: %r" % (i + 1, path, line))
            out.append(norm_path(img))
    return out


def identity(p):
    name = p.split("\\")[-1]
    return name[len("synth_"):].split("__")[0] if name.startswith("synth_") else name


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="folder with scores_f32.bin, probes.tsv, gallery.tsv")
    ap.add_argument("--pairs", required=True, help="pairs_rel.csv.gz (image_a = reference, image_b = probe)")
    ap.add_argument("--out", default="external_tnorm_scores.csv.gz")
    a = ap.parse_args()
    probes = read_list(os.path.join(a.matrix, "probes.tsv"))
    gallery = read_list(os.path.join(a.matrix, "gallery.tsv"))
    P, G = len(probes), len(gallery)
    log("matrix %d probes x %d gallery" % (P, G))
    M = np.memmap(os.path.join(a.matrix, "scores_f32.bin"), dtype="<f4", mode="r", shape=(P, G))
    # cohort: every k-th gallery reference with valid scores (checked on the first 2000 probes)
    valid = np.isfinite(np.asarray(M[:2000])).mean(0) > 0.5
    cand = np.where(valid)[0]
    cohort = cand[np.linspace(0, len(cand) - 1, N_COHORT).astype(int)]
    c_ids = np.array([identity(gallery[j]) for j in cohort])
    log("cohort: %d references" % len(cohort))
    g_index = {g: j for j, g in enumerate(gallery)}
    p_index = {p: i for i, p in enumerate(probes)}
    # per-probe statistics, in blocks of rows
    mu = np.full(P, np.nan, np.float32); sd = mu.copy(); mu_k = mu.copy(); sd_k = mu.copy()
    p_ids = np.array([identity(p) for p in probes])
    for s in range(0, P, 5000):
        S = logit(np.asarray(M[s:s + 5000][:, cohort], dtype=np.float64))
        S[p_ids[s:s + 5000, None] == c_ids[None, :]] = np.nan          # own identity out
        S[~np.isfinite(S)] = np.nan
        mu[s:s + 5000] = np.nanmean(S, 1); sd[s:s + 5000] = np.nanstd(S, 1)
        T = -np.sort(-np.where(np.isnan(S), -1e9, S), axis=1)[:, :TOPK]
        mu_k[s:s + 5000] = T.mean(1); sd_k[s:s + 5000] = T.std(1)
        if (s // 5000) % 10 == 0:
            log("cohort statistics %d/%d" % (min(s + 5000, P), P))
    sd = np.maximum(sd, 1e-6); sd_k = np.maximum(sd_k, 1e-6)
    # the pairs
    n = miss = 0
    opener = gzip.open if a.pairs.endswith(".gz") else io.open
    with opener(a.pairs, "rt", encoding="utf-8") as fin, gzip.open(a.out, "wt", encoding="utf-8", newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.writer(fout)
        w.writerow(["image_a", "image_b", "score_raw", "score_tnorm", "score_top100"])
        for row in r:
            ia, ib = g_index.get(norm_path(row["image_a"])), p_index.get(norm_path(row["image_b"]))
            if ia is None or ib is None:
                miss += 1; continue
            s = float(M[ib, ia])
            if not np.isfinite(s) or not np.isfinite(mu[ib]):
                w.writerow([row["image_a"], row["image_b"], "", "", ""]); continue
            ls = float(logit(np.array([s]))[0])
            w.writerow([row["image_a"], row["image_b"], "%.6f" % s,
                        "%.5f" % ((ls - mu[ib]) / sd[ib]), "%.5f" % ((ls - mu_k[ib]) / sd_k[ib])])
            n += 1
    log("written %d scored pairs to %s (%d pairs not in the matrix)" % (n, a.out, miss))
    if os.path.getsize(a.out) > 30e6:
        log("WARNING: output is larger than 30 MB")


if __name__ == "__main__":
    main()
