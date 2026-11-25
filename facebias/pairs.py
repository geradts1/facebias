"""
Build comparison pairs from the image metadata produced by generate.py
(or from any real dataset with the same columns).

Forensic set-up: a *reference* image (good quality, e.g. custody photo) is
compared with a *questioned* image (e.g. CCTV-like condition).
  - mated:     reference and questioned image of the same identity
  - non-mated: different identities; sampled both within the same demographic
               group (relevant impostors) and across groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEMO_COLS = ["sex", "age_bin", "ethnicity"]
EXTRA_COLS = ["gender_presentation", "age_band", "ancestry", "headwear", "monk_skin_tone",
              "glasses", "facial_hair", "edit", "ofiq_unified", "face_width_px"]


def demo_key(df: pd.DataFrame, cols=DEMO_COLS) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def build_pairs(meta: pd.DataFrame, reference_condition="reference",
                probe_conditions=None, n_nonmated_same=20, n_nonmated_cross=5,
                seed=0, demo_cols=None) -> pd.DataFrame:
    """demo_cols: attributes that define 'same demographic group' for the
    non-mated sampling (default sex, age_bin, ethnicity; with many groups use
    a coarser key, e.g. ['sex', 'ethnicity'])."""
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    meta["demo"] = demo_key(meta, demo_cols or DEMO_COLS)
    refs = meta[meta.condition == reference_condition]
    probes = meta[meta.condition != reference_condition]
    if probe_conditions:
        probes = probes[probes.condition.isin(probe_conditions)]
    probes_by_demo = {k: g for k, g in probes.groupby("demo")}
    carry = DEMO_COLS + ["demo"] + [c for c in EXTRA_COLS if c in meta.columns]

    rows = []

    def add(r, p, mated):
        rows.append({
            "image_a": r.path, "image_b": p.path, "id_a": r.identity_id, "id_b": p.identity_id,
            "mated": int(mated), "condition_b": p.condition,
            **{f"{c}_a": r[c] for c in carry},
            **{f"{c}_b": p[c] for c in carry},
            "same_demo": r.demo == p.demo,
        })

    for _, r in refs.iterrows():
        # mated
        for _, p in probes[probes.identity_id == r.identity_id].iterrows():
            add(r, p, True)
        # non-mated, same demographic group
        pool = probes_by_demo.get(r.demo)
        if pool is not None:
            pool = pool[pool.identity_id != r.identity_id]
            for i in rng.choice(len(pool), min(n_nonmated_same, len(pool)), replace=False):
                add(r, pool.iloc[i], False)
        # non-mated, other groups
        other = probes[(probes.demo != r.demo)]
        for i in rng.choice(len(other), min(n_nonmated_cross, len(other)), replace=False):
            add(r, other.iloc[i], False)

    return pd.DataFrame(rows)


CARRY_EXTRA = ["w_balance", "in_balanced_subset"]


def build_pairs_stratified(meta: pd.DataFrame, reference_condition="reference",
                           n_same_per_cond=4, n_cross_per_cond=2, demo_cols=None,
                           seed=0, exclude_conditions=()) -> pd.DataFrame:
    """Pairs for LR calibration and bias analysis, stratified per probe condition.

    For every reference image (A side):
      - mated:     all probes of the same identity (every condition),
      - non-mated: per probe condition `n_same_per_cond` probes from other
                   identities in the same demographic group (demo_cols) and
                   `n_cross_per_cond` probes from other groups.
    Every (group, condition) cell thus gets non-mated scores in proportion to
    its number of references, so per-condition / per-group calibration is
    possible. Output has the same columns as build_pairs (plus w_balance,
    in_balanced_subset and edit carried as *_a/*_b)."""
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    demo_cols = demo_cols or ["gender_presentation", "ancestry"]
    meta["demo"] = demo_key(meta, demo_cols)
    refs = meta[meta.condition == reference_condition].reset_index(drop=True)
    probes = meta[(meta.condition != reference_condition)
                  & ~meta.condition.isin(exclude_conditions)].reset_index(drop=True)
    carry = [c for c in DEMO_COLS + ["demo"] + EXTRA_COLS + CARRY_EXTRA if c in meta.columns]

    ia, ib = [], []
    # mated
    pid = probes.groupby("identity_id").indices
    for i, r in enumerate(refs.identity_id):
        idx = pid.get(r, [])
        ia += [i] * len(idx); ib += list(idx)
    # non-mated, per condition
    for cond, pc in probes.groupby("condition"):
        by_demo = {d: g.index.values for d, g in pc.groupby("demo")}
        all_idx = pc.index.values
        p_id = probes.identity_id.values; p_demo = probes.demo.values
        for i, (rid, rdemo) in enumerate(zip(refs.identity_id, refs.demo)):
            pool = by_demo.get(rdemo, np.array([], int))
            pool = pool[p_id[pool] != rid]
            k = min(n_same_per_cond, len(pool))
            if k:
                sel = rng.choice(pool, k, replace=False); ia += [i] * k; ib += list(sel)
            k = n_cross_per_cond
            if k:
                # rejection sampling from the whole condition pool (cheap, groups are many)
                got = []
                for _ in range(50):
                    c = rng.choice(all_idx, k * 2)
                    got += [x for x in c if p_demo[x] != rdemo and x not in got]
                    if len(got) >= k:
                        break
                got = got[:k]; ia += [i] * len(got); ib += got
    A = refs.iloc[ia].reset_index(drop=True)
    B = probes.iloc[ib].reset_index(drop=True)
    out = pd.DataFrame({"image_a": A.path, "image_b": B.path, "id_a": A.identity_id,
                        "id_b": B.identity_id,
                        "mated": (A.identity_id.values == B.identity_id.values).astype(int),
                        "condition_b": B.condition})
    for c in carry:
        out[f"{c}_a"] = A[c].values; out[f"{c}_b"] = B[c].values
    out["same_demo"] = A.demo.values == B.demo.values
    out = out.drop_duplicates(["image_a", "image_b"]).reset_index(drop=True)
    out.insert(0, "pair_id", [f"p{i:07d}" for i in range(len(out))])
    return out
