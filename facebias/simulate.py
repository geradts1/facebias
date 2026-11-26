"""Simulated System C-like scores with a known, injected bias - to test the
pipeline end-to-end without GPU, images or a System C licence."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .pairs import build_pairs


def simulate(n_per_cell=15, seed=0, bias=None):
    """bias: {ethnicity: (nonmated_shift, mated_shift)} on the latent logit scale.
    Default: 'East Asian' higher impostor scores (more false matches),
    'West African' lower mated scores (more false non-matches)."""
    bias = bias or {"East Asian": (1.0, 0.0), "West African": (0.0, -1.0)}
    rng = np.random.default_rng(seed)
    rows = []
    for sex, age_bin, eth in itertools.product(["woman", "man"], ["18-30", "46-60"],
                                               ["West African", "East Asian", "Northern European"]):
        for k in range(n_per_cell):
            iid = f"{sex[0]}_{age_bin}_{eth.replace(' ', '')}_{k:03d}"
            for cond in ["reference", "hq", "cctv_low", "cctv_dark"]:
                rows.append({"identity_id": iid, "sex": sex, "age_bin": age_bin, "ethnicity": eth,
                             "condition": cond, "path": f"{iid}/{cond}.png"})
    meta = pd.DataFrame(rows)
    pairs = build_pairs(meta, probe_conditions=["hq", "cctv_low", "cctv_dark"],
                        n_nonmated_same=25, n_nonmated_cross=8, seed=seed)

    cond_q = {"hq": 1.0, "cctv_low": 0.55, "cctv_dark": 0.65}
    q = pairs.condition_b.map(cond_q).values
    nm_shift = pairs.ethnicity_a.map(lambda e: bias.get(e, (0, 0))[0]).values
    m_shift = pairs.ethnicity_a.map(lambda e: bias.get(e, (0, 0))[1]).values
    same_demo = pairs.same_demo.values.astype(float)
    mated = pairs.mated.values == 1
    # identity-level random effect (some people are 'goats'/'lambs')
    ids = pairs.id_a.unique()
    eff = dict(zip(ids, rng.normal(0, 0.5, len(ids))))
    ide = pairs.id_a.map(eff).values
    latent = np.where(
        mated,
        -2.0 + 6.0 * q + m_shift + ide + rng.normal(0, 1.5, len(pairs)),
        -5.0 + 1.2 * same_demo + nm_shift * same_demo + 0.3 * ide + rng.normal(0, 1.0, len(pairs)),
    )
    pairs["score"] = 1 / (1 + np.exp(-latent))
    return meta, pairs
