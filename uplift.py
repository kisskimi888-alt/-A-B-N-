"""Uplift (heterogeneous treatment effect) models and Qini-based evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def gbm() -> HistGradientBoostingClassifier:
    # No early stopping -> fully deterministic; strong leaf regularisation because uplift is a
    # small signal on top of a noisy baseline.
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


class SLearner:
    """One model f(X, T); uplift = f(X, 1) - f(X, 0)."""
    name = "S-learner (GBM)"

    def __init__(self, base=None):
        self.base = base if base is not None else gbm()

    def fit(self, X, t, y):
        self.m = clone(self.base).fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X):
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return (self.m.predict_proba(np.column_stack([X, ones]))[:, 1]
                - self.m.predict_proba(np.column_stack([X, zeros]))[:, 1])


class TLearner:
    """Separate models per arm; uplift = mu1(X) - mu0(X)."""

    def __init__(self, base=None, name="T-learner (GBM)"):
        self.base = base if base is not None else gbm()
        self.name = name

    def fit(self, X, t, y):
        self.m1 = clone(self.base).fit(X[t == 1], y[t == 1])
        self.m0 = clone(self.base).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class ClassTransformation:
    """Jaskowski & Jaroszewicz (2012): Z = TY + (1-T)(1-Y); uplift = 2 P(Z=1|X) - 1.

    Valid when P(T=1) = 0.5, which holds here (~50/50 after dropping the other arm).
    """
    name = "Class transformation (GBM)"

    def __init__(self, base=None):
        self.base = base if base is not None else gbm()

    def fit(self, X, t, y):
        z = (t * y + (1 - t) * (1 - y)).astype(int)
        self.m = clone(self.base).fit(X, z)
        return self

    def predict(self, X):
        return 2 * self.m.predict_proba(X)[:, 1] - 1


class XLearner:
    """Kunzel et al. (2019) X-learner with GBM outcome models and GBM regressors for imputed effects."""
    name = "X-learner (GBM)"

    def fit(self, X, t, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                            min_samples_leaf=200, l2_regularization=1.0,
                                            early_stopping=False, random_state=0)
        m1 = clone(gbm()).fit(X[t == 1], y[t == 1])
        m0 = clone(gbm()).fit(X[t == 0], y[t == 0])
        d1 = y[t == 1] - m0.predict_proba(X[t == 1])[:, 1]
        d0 = m1.predict_proba(X[t == 0])[:, 1] - y[t == 0]
        self.tau1 = clone(reg).fit(X[t == 1], d1)
        self.tau0 = clone(reg).fit(X[t == 0], d0)
        self.p = t.mean()  # constant propensity under randomisation
        return self

    def predict(self, X):
        return self.p * self.tau0.predict(X) + (1 - self.p) * self.tau1.predict(X)


class RuleScore:
    """Non-learned baseline: score = one binary feature, with seeded random tie-breaking."""

    def __init__(self, col: int):
        self.col = col
        self.name = f"Rule (feature {col})"

    def fit(self, X, t, y):
        return self

    def predict(self, X):
        return X[:, self.col] + 1e-3 * np.random.default_rng(0).random(len(X))


def qini_curve(score, t, y, n_points: int = 101):
    """Qini curve: incremental outcomes when targeting the top-phi fraction ranked by score.

    Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k), expressed per customer in the evaluated sample.
    Returns (phi grid, Q(phi)).
    """
    order = np.argsort(-score, kind="mergesort")
    t, y = t[order], y[order]
    n = len(t)
    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    cum_yt, cum_yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(cum_c > 0, cum_yt - cum_yc * cum_t / np.maximum(cum_c, 1), 0.0)
    phi = np.linspace(0, 1, n_points)
    idx = np.clip((phi * n).astype(int) - 1, 0, n - 1)
    qv = np.where(phi == 0, 0.0, q[idx])
    return phi, qv / n


def qini_coefficient(score, t, y) -> float:
    """Area between the Qini curve and the random-targeting diagonal, normalised by total uplift Q(1).

    0 = no better than random; positive = the ranking concentrates uplift at the top.
    """
    phi, q = qini_curve(score, t, y)
    random_line = phi * q[-1]
    gap = q - random_line
    area = np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(phi))  # trapezoid rule (np.trapz is deprecated)
    return float(area / q[-1]) if q[-1] != 0 else float("nan")


def uplift_at_top(score, t, y, frac: float) -> dict:
    """Observed treatment-control difference among the top `frac` by score (and share of total uplift captured)."""
    k = int(round(frac * len(score)))
    top = np.argsort(-score, kind="mergesort")[:k]
    tt, yy = t[top], y[top]
    uplift_top = yy[tt == 1].mean() - yy[tt == 0].mean()
    ate = y[t == 1].mean() - y[t == 0].mean()
    return {"frac": frac, "uplift_top": float(uplift_top), "ate_all": float(ate),
            "share_of_total_uplift": float(uplift_top * k / (ate * len(score)))}


def uplift_by_decile(score, t, y, n_bins: int = 10):
    order = np.argsort(-score, kind="mergesort")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins), start=1):
        tt, yy = t[idx], y[idx]
        m1, m0 = yy[tt == 1].mean(), yy[tt == 0].mean()
        se = np.sqrt(m1 * (1 - m1) / (tt == 1).sum() + m0 * (1 - m0) / (tt == 0).sum())
        rows.append({"decile": b, "n": len(idx), "pred_uplift": float(score[idx].mean()),
                     "obs_uplift": float(m1 - m0), "se": float(se)})
    return rows
