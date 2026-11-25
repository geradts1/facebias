"""
Demographic bias analysis for a face comparison system (e.g. System C),
at two levels:

1. Score / decision level (NIST FRVT Part 3 style): FMR / FNMR per group at a
   global threshold, max/min ratios, Fairness Discrepancy Rate (FDR).
2. Likelihood-ratio level (forensically relevant): is a *globally* calibrated
   LR system mis-calibrated for a subgroup?  Compare per-group Cllr, Cllr_cal
   and rates of misleading evidence under a global system with those of a
   group-specific system. A large Cllr_cal for one group under the global
   system means LRs for that group are systematically over- or understated.

All group statistics come with identity-clustered bootstrap intervals,
because comparisons sharing an identity are not independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lr import cross_validated_llrs, performance


def threshold_at_fmr(scores, y, fmr: float) -> float:
    nm = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    return float(np.quantile(nm, 1 - fmr))


def error_rates(scores, y, thr):
    scores, y = np.asarray(scores), np.asarray(y).astype(int)
    fmr = np.mean(scores[y == 0] >= thr) if (y == 0).any() else np.nan
    fnmr = np.mean(scores[y == 1] < thr) if (y == 1).any() else np.nan
    return float(fmr), float(fnmr)


def dprime(scores, y):
    s, y = np.asarray(scores), np.asarray(y).astype(int)
    m, n = s[y == 1], s[y == 0]
    return float((m.mean() - n.mean()) / np.sqrt(0.5 * (m.var() + n.var())))


def fairness_discrepancy_rate(per_group: pd.DataFrame, alpha: float = 0.5) -> float:
    """de Freitas Pereira & Marcel (2021): FDR = 1 - (a*A + (1-a)*B),
    A = max FMR difference, B = max FNMR difference between groups. 1 = fair."""
    A = per_group["FMR"].max() - per_group["FMR"].min()
    B = per_group["FNMR"].max() - per_group["FNMR"].min()
    return float(1 - (alpha * A + (1 - alpha) * B))


def _bootstrap(df, stat_fn, n_boot, seed, id_col="id_a"):
    if n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    ids = df[id_col].unique()
    groups = {i: g for i, g in df.groupby(id_col)}
    vals = []
    for _ in range(n_boot):
        sample = pd.concat([groups[i] for i in rng.choice(ids, len(ids), replace=True)])
        try:
            vals.append(stat_fn(sample))
        except Exception:
            pass
    vals = np.array(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def score_level_bias(pairs: pd.DataFrame, group_col: str, fmr_target=1e-3,
                     n_boot=200, seed=0) -> tuple[pd.DataFrame, dict]:
    """Uses only same-demographic pairs (as NIST does for the main analysis)
    so non-mated pairs are 'look-alike plausible' impostors."""
    thr = threshold_at_fmr(pairs.score, pairs.mated, fmr_target)
    same = pairs[pairs.same_demo]
    rows = []
    for g, d in same.groupby(f"{group_col}_a"):
        fmr, fnmr = error_rates(d.score, d.mated, thr)
        fmr_ci = _bootstrap(d, lambda s: error_rates(s.score, s.mated, thr)[0], n_boot, seed)
        fnmr_ci = _bootstrap(d, lambda s: error_rates(s.score, s.mated, thr)[1], n_boot, seed)
        rows.append({"group": g, "n_mated": int(d.mated.sum()),
                     "n_nonmated": int((~d.mated.astype(bool)).sum()),
                     "mean_score_mated": d.score[d.mated == 1].mean(),
                     "mean_score_nonmated": d.score[d.mated == 0].mean(),
                     "d_prime": dprime(d.score, d.mated),
                     "FMR": fmr, "FMR_lo": fmr_ci[0], "FMR_hi": fmr_ci[1],
                     "FNMR": fnmr, "FNMR_lo": fnmr_ci[0], "FNMR_hi": fnmr_ci[1]})
    t = pd.DataFrame(rows).set_index("group")
    summary = {
        "threshold": thr, "fmr_target": fmr_target,
        "FMR_max/min": t.FMR.max() / max(t.FMR.min(), 1e-12),
        "FNMR_max/min": t.FNMR.max() / max(t.FNMR.min(), 1e-12),
        "FDR(alpha=0.5)": fairness_discrepancy_rate(t),
    }
    return t, summary


def cross_demographic_fmr(pairs: pd.DataFrame, group_col: str, thr: float) -> pd.DataFrame:
    """FMR matrix for non-mated pairs: rows = group of reference, cols = probe."""
    nm = pairs[pairs.mated == 0]
    return nm.assign(fm=nm.score >= thr).pivot_table(
        index=f"{group_col}_a", columns=f"{group_col}_b", values="fm", aggfunc="mean")


def lr_level_bias(pairs: pd.DataFrame, group_col: str, n_splits=5, seed=0,
                  n_boot=200, **system_kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group LR performance under (a) a global system fitted on all groups,
    (b) a group-specific system (only that group's comparisons)."""
    same = pairs[pairs.same_demo].copy()
    same["llr_global"] = cross_validated_llrs(
        same.score, same.mated, same.id_a, same.id_b, n_splits, seed, **system_kw)
    rows = []
    for g, d in same.groupby(f"{group_col}_a"):
        llr_spec = cross_validated_llrs(d.score, d.mated, d.id_a, d.id_b,
                                        n_splits, seed, **system_kw)
        pg = performance(d.llr_global.values, d.mated.values)
        ps = performance(llr_spec, d.mated.values)
        ci = _bootstrap(d.dropna(subset=["llr_global"]),
                        lambda s: performance(s.llr_global.values, s.mated.values).get("Cllr", np.nan),
                        n_boot, seed)
        row = {"group": g}
        row.update({f"global_{k}": v for k, v in pg.items()})
        row["global_Cllr_lo"], row["global_Cllr_hi"] = ci
        row.update({f"specific_{k}": v for k, v in ps.items() if k.startswith(("Cllr", "RME", "median"))})
        rows.append(row)
    return pd.DataFrame(rows).set_index("group"), same
