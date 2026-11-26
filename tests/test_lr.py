"""Cross-check our implementation against the NFI `lir` package (pip install lir)."""
import numpy as np
import pytest

from facebias.lr import elub, cllr, cllr_min, ScoreLRSystem, transform_scores

lir = pytest.importorskip("lir")
# the cross-check uses the lir >= 1.x API (LLRData; needs Python >= 3.11)
_models = pytest.importorskip("lir.data.models", reason="lir version too old for cross-check")
if not hasattr(_models, "LLRData"):
    pytest.skip("lir version too old for cross-check (needs lir >= 1.8, Python >= 3.11)",
                allow_module_level=True)


def _data(seed=0, n1=300, n0=3000):
    rng = np.random.default_rng(seed)
    llr = np.concatenate([rng.normal(1.5, 1.2, n1), rng.normal(-1.5, 1.2, n0)])
    y = np.concatenate([np.ones(n1), np.zeros(n0)]).astype(int)
    return llr, y


def _llrdata(llr, y):
    from lir.data.models import LLRData
    return LLRData(features=llr.reshape(-1, 1).copy(), hypothesis=y)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_elub_matches_lir(seed):
    from lir.algorithms.bayeserror import elub as lir_elub
    llr, y = _data(seed)
    assert np.allclose(elub(llr, y), lir_elub(_llrdata(llr, y)))


@pytest.mark.parametrize("seed", [0, 1])
def test_cllr_matches_lir(seed):
    from lir import metrics
    llr, y = _data(seed)
    assert np.isclose(cllr(llr, y), metrics.cllr(_llrdata(llr, y)))
    assert np.isclose(cllr_min(llr, y), metrics.cllr_min(_llrdata(llr, y)), atol=1e-6)


def test_logit_matches_lir():
    from lir.algorithms.logistic_regression import LogitCalibrator as L
    rng = np.random.default_rng(3)
    s = np.concatenate([rng.beta(8, 2, 200), rng.beta(1, 15, 2000)])
    y = np.concatenate([np.ones(200), np.zeros(2000)]).astype(int)
    sys_ = ScoreLRSystem("logit", "logit", use_elub=False).fit(s, y)
    x = transform_scores(s, "logit")
    ref = L().fit(_llrdata(x, y)).apply(_llrdata(x, y)).llrs
    assert np.allclose(sys_.llr(s), ref, atol=1e-3)


def test_perfect_calibration_low_cllr_cal():
    rng = np.random.default_rng(5)
    # scores generated from two Gaussians -> true LR known; logit calibration on x should be ~perfect
    x = np.concatenate([rng.normal(2, 1, 5000), rng.normal(0, 1, 5000)])
    y = np.concatenate([np.ones(5000), np.zeros(5000)]).astype(int)
    sys_ = ScoreLRSystem("logit", "none", use_elub=False).fit(x, y)
    true_llr = (2 * x - 2) / np.log(10)
    mid = np.abs(x - 1) < 2
    assert np.allclose(sys_.llr(x)[mid], true_llr[mid], atol=0.1)
