"""
Join a System C score export onto pairs_meta.csv.

    python tools/import_external_scores.py --scores fv_scores.csv [--fte fte.txt] [--version "vendor SDK x.y"]

Accepted --scores layouts (matched on file name, so absolute/relative paths do not matter):
  * long:   two path columns + a score column, e.g. image_a,image_b,score or gallery,probe,score
            (the order of the two path columns does not matter: both directions are tried)
  * matrix: first column = gallery image, other column headers = probe images  (--matrix)
--fte: optional text/CSV with images that failed enrolment (one path per line);
       pairs with such an image get an empty score and fte=True.
Writes data/scored_system_c.csv and prints coverage and FTE rates per group.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def name(p):
    return PureWindowsPath(str(p).strip().strip('"')).name


def load_scores(path, matrix=False, sep=None):
    df = pd.read_csv(path, sep=sep, engine="python")
    if matrix:
        g = df.columns[0]
        df = df.melt(id_vars=g, var_name="b", value_name="score").rename(columns={g: "a"})
    else:
        cols = [c.lower() for c in df.columns]
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        sc = next((c for c, l in zip(df.columns, cols) if l in ("score", "similarity", "sim")), None)
        sc = sc or (num[-1] if num else df.columns[2])
        paths = [c for c in df.columns if c != sc and df[c].astype(str).str.contains(r"\.(?:png|jpe?g|bmp)",
                                                                                     case=False).any()]
        if len(paths) < 2:
            raise SystemExit(f"could not find two image-path columns in {list(df.columns)}")
        df = df.rename(columns={paths[0]: "a", paths[1]: "b", sc: "score"})[["a", "b", "score"]]
    df["a"] = df.a.map(name); df["b"] = df.b.map(name)
    df["score"] = pd.to_numeric(df.score, errors="coerce")
    return df


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--pairs", default="data/system_c_export/pairs_meta.csv")
    p.add_argument("--out", default="data/scored_system_c.csv")
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--sep", default=None)
    p.add_argument("--fte", default=None)
    p.add_argument("--version", default="", help="System C version/config, stored in column 'matcher'")
    a = p.parse_args(argv)

    pairs = pd.read_csv(ROOT / a.pairs if not Path(a.pairs).is_absolute() else a.pairs)
    s = load_scores(a.scores, a.matrix, a.sep)
    lut = {}
    for x, y, v in zip(s.a, s.b, s.score):
        lut[(x, y)] = v; lut.setdefault((y, x), v)
    na, nb = pairs.image_a.map(name), pairs.image_b.map(name)
    pairs["score"] = [lut.get((x, y), np.nan) for x, y in zip(na, nb)]
    pairs["fte"] = False
    if a.fte:
        bad = {name(l) for l in Path(a.fte).read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()}
        pairs["fte"] = na.isin(bad) | nb.isin(bad)
        pairs.loc[pairs.fte, "score"] = np.nan
    pairs["matcher"] = ("system_c " + a.version).strip()
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    pairs.to_csv(out, index=False)

    cov = pairs.score.notna().mean()
    print(f"{len(pairs)} pairs, scored {cov:.1%}, FTE pairs {pairs.fte.mean():.2%} -> {out}")
    if cov < 0.99:
        miss = pairs[pairs.score.isna() & ~pairs.fte]
        print(f"WARNING: {len(miss)} pairs without score and not FTE, e.g. {miss.image_a.iloc[0]} vs {miss.image_b.iloc[0]}"
              if len(miss) else "")
    for g in ["ancestry_b", "gender_presentation_b", "age_band_b", "condition_b"]:
        if g in pairs.columns:
            t = pairs.groupby(g).agg(missing=("score", lambda x: x.isna().mean()), fte=("fte", "mean"))
            print(f"\n{g}\n" + (t * 100).round(2).to_string())
    if pairs.score.notna().any():
        print("\nscore range:", pairs.score.min(), pairs.score.max(),
              "(use --transform logit for [0,1] scores, none otherwise)")


if __name__ == "__main__":
    main()
