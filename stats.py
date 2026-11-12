"""Core A/B-testing statistics: SRM, balance, difference in means, CUPED, regression adjustment."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass
class Estimate:
    """A treatment-effect estimate on the absolute scale, plus relative lift."""
    method: str
    control_mean: float
    treatment_mean: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    rel_lift: float
    rel_ci_low: float
    rel_ci_high: float

    def to_dict(self) -> dict:
        return asdict(self)


def srm_test(counts: dict[str, int], expected_share: dict[str, float] | None = None) -> dict:
    """Chi-square sample-ratio-mismatch test (default: equal allocation)."""
    names = list(counts)
    obs = np.array([counts[k] for k in names], dtype=float)
    share = np.array([expected_share[k] for k in names]) if expected_share else np.full(len(obs), 1 / len(obs))
    chi2, p = stats.chisquare(obs, obs.sum() * share)
    return {"counts": counts, "chi2": float(chi2), "p_value": float(p)}


def standardized_mean_diff(x_t: np.ndarray, x_c: np.ndarray) -> float:
    """SMD with pooled SD; |SMD| < 0.1 is the usual 'balanced' rule of thumb."""
    pooled = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
    return float((np.mean(x_t) - np.mean(x_c)) / pooled) if pooled > 0 else 0.0


def _relative(diff: float, se_diff: float, mean_c: float, se_c: float, cov_dc: float, z: float):
    """Delta-method CI for (mean_t - mean_c) / mean_c given diff, its SE and Cov(diff, mean_c)."""
    rel = diff / mean_c
    # gradient of g(d, c) = d / c is (1/c, -d/c^2)
    var = se_diff**2 / mean_c**2 + diff**2 * se_c**2 / mean_c**4 - 2 * diff * cov_dc / mean_c**3
    se_rel = np.sqrt(max(var, 0.0))
    return rel, rel - z * se_rel, rel + z * se_rel


def diff_in_means(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05, method: str = "unadjusted") -> Estimate:
    """Welch-style difference in means with normal-approximation CI (n is large here)."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    m_t, m_c = y_t.mean(), y_c.mean()
    v_t, v_c = y_t.var(ddof=1) / len(y_t), y_c.var(ddof=1) / len(y_c)
    diff, se = m_t - m_c, np.sqrt(v_t + v_c)
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(diff / se))
    # Cov(m_t - m_c, m_c) = -Var(m_c) because arms are independent
    rel, rl, rh = _relative(diff, se, m_c, np.sqrt(v_c), -v_c, z)
    return Estimate(method, m_c, m_t, diff, se, diff - z * se, diff + z * se, p, rel, rl, rh)


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """theta = Cov(Y, X) / Var(X), estimated on the pooled sample (both arms)."""
    return float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))


def cuped(y_t, y_c, x_t, x_c, alpha: float = 0.05) -> tuple[Estimate, dict]:
    """CUPED (Deng et al. 2013) with a single pre-experiment covariate.

    Y_adj = Y - theta * (X - mean(X)). Because X is pre-treatment and assignment is random,
    E[X_t] = E[X_c] and the adjusted difference stays unbiased while variance falls by ~rho^2.
    """
    y = np.concatenate([y_t, y_c]).astype(float)
    x = np.concatenate([x_t, x_c]).astype(float)
    theta = cuped_theta(y, x)
    x_bar = x.mean()
    adj_t = np.asarray(y_t, float) - theta * (np.asarray(x_t, float) - x_bar)
    adj_c = np.asarray(y_c, float) - theta * (np.asarray(x_c, float) - x_bar)
    est = diff_in_means(adj_t, adj_c, alpha, method="cuped")
    # Report control/treatment means on the raw scale so the relative lift is interpretable.
    raw = diff_in_means(y_t, y_c, alpha)
    est.control_mean, est.treatment_mean = raw.control_mean, raw.treatment_mean
    z = stats.norm.ppf(1 - alpha / 2)
    v_c = np.asarray(y_c, float).var(ddof=1) / len(y_c)
    est.rel_lift, est.rel_ci_low, est.rel_ci_high = _relative(est.diff, est.se, raw.control_mean, np.sqrt(v_c), -v_c, z)
    rho = float(np.corrcoef(y, x)[0, 1])
    info = {
        "theta": theta,
        "corr_y_x": rho,
        "variance_reduction": float(1 - np.var(np.concatenate([adj_t, adj_c]), ddof=1) / np.var(y, ddof=1)),
    }
    return est, info


def regression_adjusted(df: pd.DataFrame, outcome: str, treat: str, covariates: list[str],
                        alpha: float = 0.05) -> Estimate:
    """Lin (2013) estimator: OLS of Y on T, centred X and T x centred X, with HC2 robust SEs.

    This is the multi-covariate generalisation of CUPED; interacting T with centred covariates
    keeps the coefficient on T equal to the ATE and never hurts asymptotic precision.
    """
    X = df[covariates].astype(float)
    Xc = X - X.mean()
    t = df[treat].astype(float).to_numpy()
    design = pd.concat([pd.Series(t, index=df.index, name="T"), Xc,
                        Xc.mul(t, axis=0).add_prefix("T_x_")], axis=1)
    design = sm.add_constant(design)
    fit = sm.OLS(df[outcome].astype(float), design).fit(cov_type="HC2")
    diff, se = float(fit.params["T"]), float(fit.bse["T"])
    z = stats.norm.ppf(1 - alpha / 2)
    y = df[outcome].to_numpy(float)
    y_t, y_c = y[t == 1], y[t == 0]
    m_c, v_c = y_c.mean(), y_c.var(ddof=1) / len(y_c)
    # Cov(ATE_hat, mean_c): approximate with the unadjusted value -Var(mean_c) (conservative enough here)
    rel, rl, rh = _relative(diff, se, m_c, np.sqrt(v_c), -v_c, z)
    return Estimate("regression_adjusted", m_c, y_t.mean(), diff, se, diff - z * se, diff + z * se,
                    float(2 * stats.norm.sf(abs(diff / se))), rel, rl, rh)


def bootstrap_diff(y_t, y_c, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI for a difference in means (robustness check for skewed spend)."""
    rng = np.random.default_rng(seed)
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = rng.choice(y_t, len(y_t)).mean() - rng.choice(y_c, len(y_c)).mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls FWER across the family)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def mde(sd: float, n_per_arm: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable absolute effect for a two-arm test with equal n and common SD."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2 / n_per_arm))
