import numpy as np
import pandas as pd

from facebias.synth.attributes import sample_identities, _plausible
from facebias.synth.prompts import reference_prompt, edit_prompt, PROBE_EDITS, HEADWEAR_DESC
from facebias.synth.qc import near_duplicate_identities, leakage_scores
from facebias.synth.build import SynthConfig, _assign_edits


def test_balanced_and_plausible():
    ids = sample_identities(1000, seed=1)
    df = pd.DataFrame([i.attrs for i in ids])
    assert df.ancestry.value_counts().max() - df.ancestry.value_counts().min() <= 1
    assert abs(df.age_band.value_counts()["18-29"] - df.age_band.value_counts()["30-44"]) <= 1
    assert all(_plausible(i.attrs) for i in ids)
    assert (df[df.gender_presentation == "woman"].facial_hair == "none").all()
    assert len({i.identity_id for i in ids}) == 1000


def test_prompts_mention_headwear():
    for i in sample_identities(200, seed=2):
        p = reference_prompt(i.attrs)
        if i["headwear"] != "none":
            assert HEADWEAR_DESC[i["headwear"]] in p
        for e in PROBE_EDITS:
            assert "same person" in edit_prompt(e, i.attrs)


def test_edit_assignment_balanced():
    cfg = SynthConfig(n_identities=500)
    ids = sample_identities(500, seed=3)
    ed = _assign_edits(cfg, ids, np.random.default_rng(0))
    assert all(len(e) == cfg.edits_per_identity and len(set(e)) == len(e) for e in ed)
    # hood_up is skipped for covering religious headwear by design
    counts = pd.Series([x for e in ed for x in e if x not in ("ref2", "hood_up")]).value_counts()
    assert counts.max() / counts.min() < 1.3


def test_dedup_and_leakage():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(50, 512)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    E[10] = E[3] + 0.01 * rng.normal(size=512)          # near duplicate
    ids = [f"i{k}" for k in range(50)]
    assert near_duplicate_identities(ids, E, 0.9) == {"i10"}
    gallery = np.vstack([rng.normal(size=(1000, 512)), E[7:8]])
    s = leakage_scores(E, gallery)
    assert s[7] > 0.99 and np.median(s) < 0.3
