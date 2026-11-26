"""
Command line:

  python -m facebias.cli generate --config config.yaml
  python -m facebias.cli pairs    --meta data/synth/metadata.csv --out data/pairs.csv
  python -m facebias.cli score    --pairs data/pairs.csv --matcher system_c --command "fvwrap.exe" --out data/pairs_scored.csv
  python -m facebias.cli evaluate --pairs data/pairs_scored.csv --group ethnicity --out results/
  python -m facebias.cli case     --score 0.83 --reference data/pairs_scored.csv --filter sex=man ethnicity="West African" condition_b=cctv_low
  python -m facebias.cli demo     --out results_demo/
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def _evaluate(pairs, group, out, calibrator, transform, fmr, n_boot, n_splits):
    from .bias import score_level_bias, lr_level_bias, cross_demographic_fmr
    from .lr import cross_validated_llrs, performance
    from .report import make_report

    n0 = len(pairs)
    pairs = pairs.dropna(subset=["score"]).copy()
    if len(pairs) < n0:
        print(f"warning: {n0 - len(pairs)} pairs without score (failure to enrol) removed; "
              f"check FTE rates per group in the raw file!")
    kw = dict(calibrator=calibrator, transform=transform)
    score_tab, score_sum = score_level_bias(pairs, group, fmr, n_boot)
    lr_tab, same = lr_level_bias(pairs, group, n_splits=n_splits, n_boot=n_boot, **kw)
    cross = cross_demographic_fmr(pairs, group, score_sum["threshold"])
    overall = performance(same.llr_global.values, same.mated.values)
    info = {**kw, "group": group, "n_pairs": len(pairs), "overall": overall}
    path = make_report(out, same, score_tab, score_sum, lr_tab, cross, group, info)
    print(score_tab.round(4).to_string())
    cols = [c for c in lr_tab.columns if c.endswith(("Cllr", "Cllr_cal", "RMED_nonmated_LR>1", "RMEP_mated_LR<1"))]
    print(lr_tab[cols].round(3).to_string())
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in overall.items()}, indent=1))
    print(f"report: {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="facebias")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate"); g.add_argument("--config", default="config.yaml")

    pr = sub.add_parser("pairs")
    pr.add_argument("--meta", required=True); pr.add_argument("--out", required=True)
    pr.add_argument("--probe-conditions", nargs="*", default=None)
    pr.add_argument("--n-same", type=int, default=20); pr.add_argument("--n-cross", type=int, default=5)
    pr.add_argument("--demo-cols", nargs="*", default=None,
                    help="attributes defining 'same group' for non-mated sampling, e.g. sex ethnicity")

    s = sub.add_parser("score")
    s.add_argument("--pairs", required=True); s.add_argument("--out", required=True)
    s.add_argument("--matcher", choices=["system_c", "scorefile", "arcface"], default="system_c")
    s.add_argument("--command", help="System C wrapper command (system_c matcher)")
    s.add_argument("--scorefile", help="CSV image_a,image_b,score (scorefile matcher)")

    e = sub.add_parser("evaluate")
    e.add_argument("--pairs", required=True); e.add_argument("--out", default="results")
    e.add_argument("--group", default="ethnicity", help="sex | age_bin | ethnicity | demo")
    e.add_argument("--calibrator", choices=["logit", "kde"], default="logit")
    e.add_argument("--transform", choices=["logit", "none", "log"], default="logit",
                   help="logit for System C [0,1] scores; none for cosine scores")
    e.add_argument("--fmr", type=float, default=1e-3)
    e.add_argument("--n-boot", type=int, default=200)
    e.add_argument("--n-splits", type=int, default=5)

    c = sub.add_parser("case")
    c.add_argument("--score", type=float, required=True)
    c.add_argument("--reference", required=True, help="scored pairs CSV used for calibration")
    c.add_argument("--filter", nargs="*", default=[], help="col=value, selects case-relevant reference data")
    c.add_argument("--calibrator", choices=["logit", "kde"], default="logit")
    c.add_argument("--transform", choices=["logit", "none", "log"], default="logit")

    d = sub.add_parser("demo"); d.add_argument("--out", default="results_demo")
    d.add_argument("--group", default="ethnicity"); d.add_argument("--n-boot", type=int, default=100)

    a = p.parse_args(argv)

    if a.cmd == "generate":
        from .generate import GenConfig, generate
        generate(GenConfig.from_yaml(a.config))

    elif a.cmd == "pairs":
        from .pairs import build_pairs
        pairs = build_pairs(pd.read_csv(a.meta), probe_conditions=a.probe_conditions,
                            n_nonmated_same=a.n_same, n_nonmated_cross=a.n_cross,
                            demo_cols=a.demo_cols)
        pairs.to_csv(a.out, index=False)
        print(f"{len(pairs)} pairs ({pairs.mated.sum()} mated) -> {a.out}")

    elif a.cmd == "score":
        from .matchers import get_matcher
        pairs = pd.read_csv(a.pairs)
        kw = {"system_c": {"command": a.command}, "scorefile": {"path": a.scorefile},
              "arcface": {}}[a.matcher]
        pairs["score"] = get_matcher(a.matcher, **kw).score_pairs(pairs)
        pairs["matcher"] = a.matcher
        pairs.to_csv(a.out, index=False)
        print(f"scored -> {a.out}; missing scores: {pairs.score.isna().sum()}")

    elif a.cmd == "evaluate":
        _evaluate(pd.read_csv(a.pairs), a.group, a.out, a.calibrator, a.transform,
                  a.fmr, a.n_boot, a.n_splits)

    elif a.cmd == "case":
        from .lr import ScoreLRSystem, cross_validated_llrs, performance, verbal
        ref = pd.read_csv(a.reference).dropna(subset=["score"])
        for f in a.filter:
            # 'sex=man' -> both persons in the pair are 'man'; 'condition_b=cctv_low' -> exact column
            col, val = f.split("=", 1)
            cols = [col] if col in ref.columns else [f"{col}_a", f"{col}_b"]
            for cc in cols:
                ref = ref[ref[cc].astype(str) == val]
        if ref.mated.nunique() < 2:
            raise SystemExit("reference selection contains no mated or no non-mated pairs")
        sys_ = ScoreLRSystem(a.calibrator, a.transform).fit(ref.score, ref.mated)
        llr = float(sys_.llr([a.score])[0]); raw = float(sys_.raw_llr([a.score])[0])
        cv = performance(cross_validated_llrs(ref.score, ref.mated, ref.id_a, ref.id_b,
                                              calibrator=a.calibrator, transform=a.transform),
                         ref.mated.values)
        print(json.dumps({
            "case_score": a.score, "filters": a.filter,
            "n_reference_mated": sys_.n_mated, "n_reference_nonmated": sys_.n_nonmated,
            "elub_log10_bounds": [sys_.lower_bound, sys_.upper_bound],
            "log10_LR_unbounded": round(raw, 3), "log10_LR_reported": round(llr, 3),
            "LR_reported": float(f"{10 ** llr:.3g}"),
            "verbal": verbal(llr), "validation_cv": cv}, indent=1, default=float))

    elif a.cmd == "demo":
        from .simulate import simulate
        meta, pairs = simulate()
        _evaluate(pairs, a.group, a.out, "logit", "logit", 1e-3, a.n_boot, 5)


if __name__ == "__main__":
    main()
