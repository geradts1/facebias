"""Re-run the per-condition and LR-direction analysis of the open matchers on exactly the pairs
System C could score (pairs with an image in system_c_fte.txt removed), so the systems are compared
like for like.

    python tools/common_subset.py --fte <system_c_fte.txt>
Output: results_by_condition_<m>_fvsubset.*, results_lr_direction_<m>_fvsubset.*
"""
import argparse
import sys
from pathlib import Path, PureWindowsPath

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from analyze_by_condition import analyse  # noqa: E402
from lr_direction import direction, report  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fte", required=True)
    p.add_argument("--models", nargs="+", default=["adaface", "arcface", "adaface_vit", "arcface_ir101", "adaface_ms1mv2"])
    a = p.parse_args()
    bad = {PureWindowsPath(l.strip()).name for l in open(a.fte, encoding="utf-8", errors="ignore") if l.strip()}
    print(f"{len(bad)} images without a System C template", flush=True)
    for m in a.models:
        src = ROOT / f"data/scored_{m}_v2.csv"
        if not src.exists():
            continue
        d = pd.read_csv(src)
        na = d.image_a.map(lambda s: PureWindowsPath(str(s)).name)
        nb = d.image_b.map(lambda s: PureWindowsPath(str(s)).name)
        d = d[~(na.isin(bad) | nb.isin(bad))].reset_index(drop=True)
        name = f"{m}_fvsubset"
        print(f"{m}: {len(d)} pairs left", flush=True)
        analyse(d, name)
        report(direction(d, name, "none"), name)
    print("KLAAR", flush=True)


if __name__ == "__main__":
    main()
