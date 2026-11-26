"""
Score-based likelihood ratios following the NFI approach.

Pipeline (as used by the Netherlands Forensic Institute and implemented in its
open-source `lir` package):

    raw comparison score  ->  (optional) score transform
                          ->  calibration  (logistic regression or KDE)
                          ->  ELUB bounding (empirical lower / upper bound)
                          ->  log10 LR

Calibration data must be *case-relevant*: comparisons between images whose
conditions (quality, pose, resolution, demographics of the persons involved)
resemble the questioned/reference material in the case.  Validation is done
with identity-disjoint cross-validation and Cllr / Cllr_min / Cllr_cal,
rates of misleading evidence, Tippett and ECE plots.

References
----------
- Vergeer, P., van Es, A., de Jongh, A., Alberink, I., Stoel, R. (2016).
  Numerical likelihood ratios outputted by LR systems are often based on
  extrapolation: when to stop extrapolating? Science & Justice 56, 482-491.
  (ELUB; the implementation below is a port of `lir.algorithms.bayeserror.elub`)
- Brümmer, N., du Preez, J. (2006). Application-independent evaluation of
  speaker detection. Computer Speech & Language 20, 230-275. (Cllr)
- Meuwly, D., Ramos, D., Haraksim, R. (2017). A guideline for the validation
  of likelihood ratio methods used for forensic evidence evaluation.
  Forensic Science International 276, 142-153.
- NFI lir package: https://github.com/NetherlandsForensicInstitute/lir
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

LN10 = np.log(10.0)


# --------------------------------------------------------------------------
# Score transforms
# --------------------------------------------------------------------------
def transform_scores(scores: np.ndarray, kind: str = "logit", eps: float = 1e-6) -> np.ndarray:
    """System C similarity scores live in [0, 1] and pile up near 0 and 1;
    a logit transform makes them far better behaved for calibration."""
    s = np.asarray(scores, dtype=float)
    if kind == "none":
        return s
    if kind == "logit":
        s = np.clip(s, eps, 1 - eps)
        return np.log(s / (1 - s))
    if kind == "log":
        return np.log(np.clip(s, eps, None))
    raise ValueError(f"unknown transform {kind!r}")


# --------------------------------------------------------------------------
# Calibrators: map (transformed) score -> log10 LR
# --------------------------------------------------------------------------
class LogitCalibrator:
    """Logistic regression with balanced class weights => output is an LR
    (prior odds of the training set are removed). Same as lir.LogitCalibrator,
    including sklearn's default L2 regularisation C=1.0. Note: with (nearly)
    separable scores the slope - and hence the extreme LRs - depends strongly
    on C; ELUB bounding guards against the resulting over-extrapolation."""

    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(class_weight="balanced", C=C, max_iter=1000)

    def fit(self, x, y):
        self.model.fit(np.asarray(x).reshape(-1, 1), np.asarray(y))
        return self

    def llr(self, x):
        x = np.asarray(x, dtype=float)
        ln_lr = self.model.intercept_[0] + self.model.coef_[0][0] * x
        return ln_lr / LN10


class KDECalibrator:
    """Gaussian KDE per hypothesis; LR = f(x|Hp) / f(x|Hd)."""

    def __init__(self, bw_method="silverman", floor: float = 1e-300):
        self.bw_method, self.floor = bw_method, floor

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y)
        self.kde1 = gaussian_kde(x[y == 1], bw_method=self.bw_method)
        self.kde0 = gaussian_kde(x[y == 0], bw_method=self.bw_method)
        return self

    def llr(self, x):
        x = np.asarray(x, float)
        f1 = np.maximum(self.kde1(x), self.floor)
        f0 = np.maximum(self.kde0(x), self.floor)
        return np.log10(f1) - np.log10(f0)


CALIBRATORS = {"logit": LogitCalibrator, "kde": KDECalibrator}


# --------------------------------------------------------------------------
# ELUB (port of lir.algorithms.bayeserror.elub, Vergeer et al. 2016)
# --------------------------------------------------------------------------
def _expected_utility(lrs, y, threshold_lrs, add_misleading=0):
    m_accept = lrs.reshape(-1, 1) > threshold_lrs.reshape(1, -1)
    if add_misleading > 0:
        n = len(threshold_lrs)
        m_accept = np.concatenate(
            [m_accept, np.zeros((add_misleading, n)), np.ones((add_misleading, n))])
        y = np.concatenate([y, np.ones(add_misleading), np.zeros(add_misleading)])
    return (1 - np.average(m_accept[y == 1], axis=0)
            + threshold_lrs * np.average(m_accept[y == 0], axis=0))


def elub(llrs, y, add_misleading: int = 1, step_size: float = 0.01,
         substitute_extremes=(-9.0, 9.0)) -> tuple[float, float]:
    """Empirical lower and upper bound (log10) for an LR system."""
    llrs = np.clip(np.asarray(llrs, float).copy(), *substitute_extremes)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError("ELUB needs both mated (1) and non-mated (0) comparisons")
    llr_min = max(llrs.min(), -np.log10(y.sum() + 1))
    llr_max = min(llrs.max(), np.log10((1 - y).sum() + 1))
    s_min = min(0, int(np.floor_divide(llr_min, step_size)))
    s_max = max(0, int(np.floor_divide(llr_max, step_size)) + 1)
    thr = np.linspace(s_min * step_size, s_max * step_size, s_max - s_min + 1)
    eu_neutral = _expected_utility(np.ones(len(llrs)), y, 10 ** thr)
    eu_system = _expected_utility(10 ** llrs, y, 10 ** thr, add_misleading)
    ratio = eu_neutral / eu_system
    left = thr[(thr <= 0) & (ratio < 1)]
    right = thr[(thr >= 0) & (ratio < 1)]
    lower = np.max(left + step_size, initial=thr.min())
    upper = np.min(right - step_size, initial=thr.max())
    return float(min(lower, 0)), float(max(upper, 0))


# --------------------------------------------------------------------------
# Full LR system
# --------------------------------------------------------------------------
@dataclass
class ScoreLRSystem:
    calibrator: str = "logit"
    transform: str = "logit"
    use_elub: bool = True
    lower_bound: float | None = field(default=None, init=False)
    upper_bound: float | None = field(default=None, init=False)

    def fit(self, scores, y):
        x = transform_scores(scores, self.transform)
        y = np.asarray(y).astype(int)
        self._cal = CALIBRATORS[self.calibrator]().fit(x, y)
        if self.use_elub:
            self.lower_bound, self.upper_bound = elub(self._cal.llr(x), y)
        self.n_mated, self.n_nonmated = int(y.sum()), int((1 - y).sum())
        return self

    def raw_llr(self, scores):
        return self._cal.llr(transform_scores(scores, self.transform))

    def llr(self, scores):
        out = self.raw_llr(scores)
        if self.use_elub:
            out = np.clip(out, self.lower_bound, self.upper_bound)
        return out


def identity_disjoint_folds(id_a, id_b, n_splits=5, seed=0):
    """Yield (train_mask, test_mask) such that no identity in a test pair
    occurs anywhere in the training pairs. Pairs that straddle two folds are
    only ever used for training (never tested) - this avoids optimistic
    validation caused by identity leakage."""
    id_a, id_b = np.asarray(id_a), np.asarray(id_b)
    ids = np.unique(np.concatenate([id_a, id_b]))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(ids), np.arange(len(ids)) % n_splits))
    fa = np.array([fold_of[i] for i in id_a])
    fb = np.array([fold_of[i] for i in id_b])
    for k in range(n_splits):
        test = (fa == k) & (fb == k)
        train = (fa != k) & (fb != k)
        yield train, test


def cross_validated_llrs(scores, y, id_a, id_b, n_splits=5, seed=0, n_repeats=3,
                         **system_kw):
    """Out-of-fold log10 LRs with identity-disjoint folds. Because a non-mated
    pair is only tested when both identities fall in the same fold, the CV is
    repeated with different fold assignments and the out-of-fold LRs are
    averaged; pairs never tested get NaN."""
    scores, y = np.asarray(scores, float), np.asarray(y).astype(int)
    acc, cnt = np.zeros(len(scores)), np.zeros(len(scores))
    for r in range(n_repeats):
        for train, test in identity_disjoint_folds(id_a, id_b, n_splits, seed + r):
            if test.sum() == 0 or len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 1:
                continue
            sys_ = ScoreLRSystem(**system_kw).fit(scores[train], y[train])
            acc[test] += sys_.llr(scores[test])
            cnt[test] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


# --------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------
def cllr(llrs, y) -> float:
    llrs, y = np.asarray(llrs, float), np.asarray(y).astype(int)
    lr = 10.0 ** llrs
    with np.errstate(divide="ignore", over="ignore"):
        p1 = np.mean(np.log2(1 + 1 / lr[y == 1]))
        p0 = np.mean(np.log2(1 + lr[y == 0]))
    return float(0.5 * (p1 + p0))


def pav_llrs(llrs, y) -> np.ndarray:
    """Optimally calibrated (PAV / isotonic) log10 LRs on the same data."""
    llrs, y = np.asarray(llrs, float), np.asarray(y).astype(int)
    p = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit_transform(llrs, y)
    prior_odds = y.sum() / (len(y) - y.sum())
    with np.errstate(divide="ignore"):
        return np.log10(p) - np.log10(1 - p) - np.log10(prior_odds)


def cllr_min(llrs, y) -> float:
    return cllr(pav_llrs(llrs, y), y)


def misleading_rates(llrs, y) -> dict:
    llrs, y = np.asarray(llrs, float), np.asarray(y).astype(int)
    return {
        "RMEP_mated_LR<1": float(np.mean(llrs[y == 1] < 0)),
        "RMED_nonmated_LR>1": float(np.mean(llrs[y == 0] > 0)),
        "nonmated_LR>100": float(np.mean(llrs[y == 0] > 2)),
        "mated_LR<0.01": float(np.mean(llrs[y == 1] < -2)),
    }


def performance(llrs, y) -> dict:
    ok = ~np.isnan(llrs)
    llrs, y = np.asarray(llrs)[ok], np.asarray(y).astype(int)[ok]
    if len(np.unique(y)) < 2:
        return {"n_mated": int(y.sum()), "n_nonmated": int((1 - y).sum())}
    c, cm = cllr(llrs, y), cllr_min(llrs, y)
    return {
        "n_mated": int(y.sum()), "n_nonmated": int((1 - y).sum()),
        "Cllr": c, "Cllr_min": cm, "Cllr_cal": c - cm,
        "median_llr_mated": float(np.median(llrs[y == 1])),
        "median_llr_nonmated": float(np.median(llrs[y == 0])),
        **misleading_rates(llrs, y),
    }


def ece(llrs, y, prior_log10_odds=np.linspace(-2.5, 2.5, 101)) -> np.ndarray:
    """Empirical cross-entropy (Ramos & Gonzalez-Rodriguez) over prior log10-odds."""
    llrs, y = np.asarray(llrs, float), np.asarray(y).astype(int)
    out = []
    for lo in prior_log10_odds:
        p1 = 1 / (1 + 10 ** -lo)
        post = llrs + lo
        with np.errstate(over="ignore"):
            a = np.mean(np.log2(1 + 10 ** -post[y == 1]))
            b = np.mean(np.log2(1 + 10 ** post[y == 0]))
        out.append(p1 * a + (1 - p1) * b)
    return np.array(out)


# --------------------------------------------------------------------------
# Reporting language
# --------------------------------------------------------------------------
# NFI verbal scale (check against the current NFI 'waarschijnlijkheidstermen'
# guideline before using in reports).
VERBAL_SCALE = [
    (1, 2, "ongeveer even waarschijnlijk"),
    (2, 10, "iets waarschijnlijker"),
    (10, 100, "waarschijnlijker"),
    (100, 10_000, "veel waarschijnlijker"),
    (10_000, 1_000_000, "zeer veel waarschijnlijker"),
    (1_000_000, np.inf, "extreem veel waarschijnlijker"),
]


def verbal(llr: float) -> str:
    lr = 10 ** abs(llr)
    hyp = "Hp (zelfde persoon)" if llr >= 0 else "Hd (verschillende personen)"
    for lo, hi, term in VERBAL_SCALE:
        if lo <= lr < hi:
            return f"De bevindingen zijn {term} onder {hyp} dan onder de alternatieve hypothese"
    return "?"
