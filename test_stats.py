"""Sanity tests on synthetic data where the true answer is known."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import stats as st  # noqa: E402
import uplift as up  # noqa: E402


def synthetic(n=40_000, rho=0.7, effect=0.1, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n) + effect * t
    return x, t, y


def test_diff_in_means_recovers_effect():
    x, t, y = synthetic()
    est = st.diff_in_means(y[t == 1], y[t == 0])
    assert est.ci_low < 0.1 < est.ci_high


def test_cuped_variance_reduction_matches_rho_squared():
    x, t, y = synthetic(rho=0.7)
    est, info = st.cuped(y[t == 1], y[t == 0], x[t == 1], x[t == 0])
    base = st.diff_in_means(y[t == 1], y[t == 0])
    assert abs(info["variance_reduction"] - 0.49) < 0.02
    assert abs((est.se / base.se) ** 2 - 0.51) < 0.02
    assert est.ci_low < 0.1 < est.ci_high


def test_regression_adjustment_matches_cuped_with_one_covariate():
    x, t, y = synthetic(rho=0.5)
    df = pd.DataFrame({"y": y, "t": t, "x": x})
    reg = st.regression_adjusted(df, "y", "t", ["x"])
    cup, _ = st.cuped(y[t == 1], y[t == 0], x[t == 1], x[t == 0])
    assert abs(reg.diff - cup.diff) < 0.005
    assert abs(reg.se / cup.se - 1) < 0.02


def test_holm_is_monotone_and_bounded():
    adj = st.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == [0.04, 0.09, 0.09, 0.5]


def test_srm_detects_imbalance():
    assert st.srm_test({"a": 5000, "b": 5000})["p_value"] > 0.5
    assert st.srm_test({"a": 5200, "b": 4800})["p_value"] < 0.001


def test_qini_perfect_vs_random_ranking():
    rng = np.random.default_rng(1)
    n = 20_000
    responder = rng.random(n) < 0.3          # only these customers react to treatment
    t = rng.integers(0, 2, n)
    y = ((rng.random(n) < 0.05) | (responder & (t == 1) & (rng.random(n) < 0.5))).astype(int)
    oracle = responder.astype(float) + 1e-6 * rng.random(n)
    assert up.qini_coefficient(oracle, t, y) > 0.2
    assert abs(up.qini_coefficient(rng.random(n), t, y)) < 0.1
