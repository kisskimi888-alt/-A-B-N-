"""End-to-end pipeline: randomisation checks -> A/B tests -> CUPED / regression adjustment ->
A/A calibration -> power -> subgroups -> uplift models. Writes results/ and figures/.

Usage:  python data/download.py && python run_all.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import plots  # noqa: E402
import stats as st  # noqa: E402
import uplift as up  # noqa: E402

SEED = 42
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
CONTROL, ARMS = "No E-Mail", ["Mens E-Mail", "Womens E-Mail"]
METRICS = ["visit", "conversion", "spend"]
PRE_COVARIATES = ["recency", "history", "mens", "womens", "newbie",
                  "zip_Urban", "zip_Rural", "channel_Web", "channel_Phone"]


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "hillstrom.csv")
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})  # typo in the source file
    for z in ["Urban", "Rural"]:
        df[f"zip_{z}"] = (df["zip_code"] == z).astype(int)
    for c in ["Web", "Phone"]:
        df[f"channel_{c}"] = (df["channel"] == c).astype(int)  # baseline = Multichannel
    df["buyer_type"] = np.select([(df.mens == 1) & (df.womens == 1), df.mens == 1],
                                 ["both", "mens_only"], "womens_only")
    return df


def pair(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    d = df[df.segment.isin([arm, CONTROL])].copy()
    d["treat"] = (d.segment == arm).astype(int)
    return d


def randomisation_checks(df):
    srm = st.srm_test(df.segment.value_counts().to_dict())
    balance = []
    for arm in ARMS:
        d = pair(df, arm)
        for c in PRE_COVARIATES:
            balance.append({"arm": arm, "covariate": c,
                            "smd": st.standardized_mean_diff(d.loc[d.treat == 1, c], d.loc[d.treat == 0, c])})
    return srm, balance


def ab_tests(df):
    rows = []
    for arm in ARMS:
        d = pair(df, arm)
        for m in METRICS:
            est = st.diff_in_means(d.loc[d.treat == 1, m], d.loc[d.treat == 0, m])
            row = {"arm": arm, "metric": m, "n_t": int(d.treat.sum()), "n_c": int((1 - d.treat).sum()), **est.to_dict()}
            if m == "spend":
                b = st.bootstrap_diff(d.loc[d.treat == 1, m], d.loc[d.treat == 0, m], n_boot=2000, seed=SEED)
                row["boot_ci_low"], row["boot_ci_high"] = b["ci_low"], b["ci_high"]
            rows.append(row)
    for r, p in zip(rows, st.holm([r["p_value"] for r in rows])):
        r["p_holm"] = p
    # Head-to-head: Mens vs Womens e-mail
    d = df[df.segment.isin(ARMS)]
    h2h = []
    for m in METRICS:
        est = st.diff_in_means(d.loc[d.segment == "Mens E-Mail", m], d.loc[d.segment == "Womens E-Mail", m])
        h2h.append({"metric": m, **est.to_dict()})
    # Spend among converters: descriptive only (conditioning on a post-treatment outcome).
    conv = df[df.conversion == 1].groupby("segment")["spend"].agg(["mean", "median", "count"]).reset_index()
    return rows, h2h, conv


def cross_fitted_prediction(d, outcome, n_splits=5):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    X, y = d[PRE_COVARIATES].to_numpy(float), d[outcome].to_numpy(float)
    pred = np.zeros(len(d))
    for tr, te in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                          min_samples_leaf=200, early_stopping=False, random_state=0)
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return pred


def variance_reduction(df):
    rows = []
    for arm in ARMS:
        d = pair(df, arm)
        t, c = d[d.treat == 1], d[d.treat == 0]
        for m in METRICS:
            base = st.diff_in_means(t[m], c[m])
            cup, info = st.cuped(t[m], c[m], t["history"], c["history"])
            reg = st.regression_adjusted(d, m, "treat", PRE_COVARIATES)
            # CUPAC: CUPED where the covariate is a cross-fitted ML prediction of Y from pre-period
            # features (fitted on both arms, treatment excluded, so it is still pre-treatment only).
            pred = cross_fitted_prediction(d, m)
            cpc, cinfo = st.cuped(t[m], c[m], pred[d.treat.to_numpy() == 1], pred[d.treat.to_numpy() == 0])
            cpc.method = "cupac"
            for est, extra in [(base, {}), (cup, info), (reg, {}), (cpc, cinfo)]:
                rows.append({"arm": arm, "metric": m, "label": f"{arm.split()[0]} / {m}",
                             **est.to_dict(), "se_ratio": est.se / base.se,
                             "variance_reduction_se": 1 - (est.se / base.se) ** 2,
                             "effective_n_multiplier": (base.se / est.se) ** 2, **extra})
    return rows


def aa_calibration(df, n_sims=1000):
    """Split the control group at random many times: a calibrated test rejects ~5% of the time."""
    rng = np.random.default_rng(SEED)
    c = df[df.segment == CONTROL].reset_index(drop=True)
    out = {}
    for m in ["visit", "spend"]:
        rej = {"unadjusted": 0, "cuped": 0, "regression_adjusted": 0}
        for _ in range(n_sims):
            fake = rng.permutation(len(c)) < len(c) // 2
            d = c.assign(treat=fake.astype(int))
            t, cc = d[d.treat == 1], d[d.treat == 0]
            rej["unadjusted"] += st.diff_in_means(t[m], cc[m]).p_value < 0.05
            rej["cuped"] += st.cuped(t[m], cc[m], t["history"], cc["history"])[0].p_value < 0.05
            rej["regression_adjusted"] += st.regression_adjusted(d, m, "treat", PRE_COVARIATES).p_value < 0.05
        out[m] = {k: v / n_sims for k, v in rej.items()}
        out[m]["n_sims"] = n_sims
        # 95% Monte-Carlo band for a true 5% rate
        out[m]["mc_band"] = [0.05 - 1.96 * np.sqrt(0.05 * 0.95 / n_sims), 0.05 + 1.96 * np.sqrt(0.05 * 0.95 / n_sims)]
    return out


def power_table(df, vr_rows):
    rows = []
    c = df[df.segment == CONTROL]
    n_per_arm = len(c)
    for m in METRICS:
        sd = c[m].std(ddof=1)
        best = max(r["variance_reduction_se"] for r in vr_rows
                   if r["metric"] == m and r["arm"] == "Mens E-Mail" and r["method"] == "regression_adjusted")
        mde_raw = st.mde(sd, n_per_arm)
        mde_adj = st.mde(sd * np.sqrt(1 - best), n_per_arm)
        rows.append({"metric": m, "control_mean": c[m].mean(), "control_sd": sd, "n_per_arm": n_per_arm,
                     "mde_abs": mde_raw, "mde_rel": mde_raw / c[m].mean(),
                     "mde_abs_adjusted": mde_adj, "mde_rel_adjusted": mde_adj / c[m].mean(),
                     "variance_reduction_used": best})
    return rows


def subgroups(df):
    """Visit uplift by past purchase type, with a treatment x subgroup interaction test."""
    import statsmodels.formula.api as smf
    rows, interactions = [], []
    for arm in ARMS:
        d = pair(df, arm)
        for g, dg in d.groupby("buyer_type"):
            est = st.diff_in_means(dg.loc[dg.treat == 1, "visit"], dg.loc[dg.treat == 0, "visit"])
            rows.append({"arm": arm, "buyer_type": g, "n": len(dg), **est.to_dict()})
        fit = smf.ols("visit ~ treat * C(buyer_type)", data=d).fit(cov_type="HC2")
        # Joint Wald test that all T x subgroup interactions are zero
        names = [n for n in fit.params.index if n.startswith("treat:")]
        R = np.zeros((len(names), len(fit.params)))
        for i, n in enumerate(names):
            R[i, list(fit.params.index).index(n)] = 1
        w = fit.wald_test(R, scalar=True)
        interactions.append({"arm": arm, "wald_chi2": float(w.statistic), "df": len(names), "p_value": float(w.pvalue)})
    # Which e-mail should each buyer type get? Mens vs Womens e-mail, head-to-head.
    h2h = []
    d = df[df.segment.isin(ARMS)]
    for g, dg in d.groupby("buyer_type"):
        for m in METRICS:
            est = st.diff_in_means(dg.loc[dg.segment == "Mens E-Mail", m], dg.loc[dg.segment == "Womens E-Mail", m])
            h2h.append({"buyer_type": g, "metric": m, "n": len(dg), "mens_minus_womens": est.diff,
                        "ci_low": est.ci_low, "ci_high": est.ci_high, "p_value": est.p_value})
    return rows, interactions, h2h


def uplift_analysis(df, arm, outcome="visit", n_splits=5, n_perm=200):
    d = pair(df, arm).reset_index(drop=True)
    X = d[PRE_COVARIATES].to_numpy(float)
    t, y = d["treat"].to_numpy(), d[outcome].to_numpy()
    models = {
        "T-learner (logistic)": lambda: up.TLearner(up.logit(), name="T-learner (logistic)"),
        "S-learner (GBM)": lambda: up.SLearner(),
        "T-learner (GBM)": lambda: up.TLearner(),
        "X-learner (GBM)": lambda: up.XLearner(),
        "Class transformation (GBM)": lambda: up.ClassTransformation(),
    }
    # Transparent baseline: rank by "has bought this e-mail's merchandise before" (ties broken at random).
    rule_col = PRE_COVARIATES.index("mens" if arm.startswith("Mens") else "womens")
    rule_name = f"Rule: past {'mens' if arm.startswith('Mens') else 'womens'} buyer"
    models[rule_name] = lambda: up.RuleScore(rule_col)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    strata = 2 * t + y
    oof = {k: np.zeros(len(d)) for k in models}
    fold_qini = {k: [] for k in models}
    for tr, te in skf.split(X, strata):
        for name, make in models.items():
            m = make().fit(X[tr], t[tr], y[tr])
            oof[name][te] = m.predict(X[te])
            fold_qini[name].append(up.qini_coefficient(oof[name][te], t[te], y[te]))
    rng = np.random.default_rng(SEED)
    null = np.array([up.qini_coefficient(rng.random(len(d)), t, y) for _ in range(n_perm)])
    summary = []
    for name in models:
        q = up.qini_coefficient(oof[name], t, y)
        summary.append({"arm": arm, "outcome": outcome, "model": name, "qini_oof": q,
                        "qini_fold_mean": float(np.mean(fold_qini[name])),
                        "qini_fold_sd": float(np.std(fold_qini[name], ddof=1)),
                        "perm_p_value": float((1 + (null >= q).sum()) / (1 + n_perm)),
                        **{f"top{int(f * 100)}_share": up.uplift_at_top(oof[name], t, y, f)["share_of_total_uplift"]
                           for f in (0.2, 0.4, 0.6)}})
    best = max((r for r in summary if not r["model"].startswith("Rule")), key=lambda r: r["qini_fold_mean"])["model"]
    # Policy check on the best model: e-mail only the top 40% vs everyone (other outcomes too).
    policy = []
    k = int(0.4 * len(d))
    top = np.zeros(len(d), bool)
    top[np.argsort(-oof[best], kind="mergesort")[:k]] = True
    for m in METRICS:
        yy = d[m].to_numpy(float)
        for label, mask in [("top 40%", top), ("bottom 60%", ~top), ("all", np.ones(len(d), bool))]:
            est = st.diff_in_means(yy[mask & (t == 1)], yy[mask & (t == 0)])
            policy.append({"arm": arm, "segment": label, "metric": m, "uplift": est.diff,
                           "ci_low": est.ci_low, "ci_high": est.ci_high, "n": int(mask.sum())})
    # Plot the selected model against the transparent rule (plus the random diagonal).
    curves = {f"{best} (selected)": up.qini_curve(oof[best], t, y),
              rule_name: up.qini_curve(oof[rule_name], t, y)}
    deciles = up.uplift_by_decile(oof[best], t, y)
    null_summary = {"mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                    "q95": float(np.quantile(null, 0.95))}
    # Who is in the top 40%? Profile vs. the rest (descriptive)
    profile = (d.assign(top40=top).groupby("top40")[["recency", "history", "mens", "womens", "newbie",
                                                     "channel_Web", "channel_Phone"]].mean().T
               .rename(columns={True: "top40", False: "rest"}).reset_index().rename(columns={"index": "feature"}))
    return summary, best, policy, curves, deciles, null_summary, profile


def main():
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = load()
    out = {"seed": SEED, "n_rows": len(df)}

    srm, balance = randomisation_checks(df)
    out["srm"] = srm
    out["max_abs_smd"] = max(abs(b["smd"]) for b in balance)
    pd.DataFrame(balance).to_csv(RESULTS / "balance.csv", index=False)

    ab, h2h, conv = ab_tests(df)
    out["ab_tests"], out["mens_vs_womens"] = ab, h2h
    pd.DataFrame(ab).to_csv(RESULTS / "ab_tests.csv", index=False)
    conv.to_csv(RESULTS / "spend_among_converters.csv", index=False)

    vr = variance_reduction(df)
    out["variance_reduction"] = vr
    pd.DataFrame(vr).to_csv(RESULTS / "variance_reduction.csv", index=False)

    out["aa_calibration"] = aa_calibration(df)
    out["power"] = power_table(df, vr)
    pd.DataFrame(out["power"]).to_csv(RESULTS / "power.csv", index=False)

    sg, inter, sg_h2h = subgroups(df)
    out["subgroups"], out["subgroup_interaction_tests"], out["subgroup_mens_vs_womens"] = sg, inter, sg_h2h
    pd.DataFrame(sg_h2h).to_csv(RESULTS / "subgroups_mens_vs_womens.csv", index=False)
    pd.DataFrame(sg).to_csv(RESULTS / "subgroups_visit.csv", index=False)

    out["uplift"] = {}
    all_summ, all_pol = [], []
    for arm in ARMS:
        summ, best, pol, curves, dec, null, prof = uplift_analysis(df, arm)
        tag = arm.split()[0].lower()
        out["uplift"][arm] = {"models": summ, "best_model": best, "policy_top40": pol,
                              "deciles_best": dec, "qini_null": null}
        all_summ += summ
        all_pol += pol
        prof.to_csv(RESULTS / f"uplift_top40_profile_{tag}.csv", index=False)
        plots.qini_plot(curves, FIGURES / f"qini_{tag}.png",
                        f"{arm} vs. No E-Mail: Qini curves (out-of-fold)")
        plots.decile_plot(dec, FIGURES / f"deciles_{tag}.png",
                          f"{arm}: uplift by predicted-uplift decile", best)
    pd.DataFrame(all_summ).to_csv(RESULTS / "uplift_models.csv", index=False)
    pd.DataFrame(all_pol).to_csv(RESULTS / "uplift_policy_top40.csv", index=False)

    plots.forest_relative_lift(ab, FIGURES / "lift_forest.png", "Relative lift of each e-mail vs. No E-Mail")
    plots.variance_reduction_plot(vr, FIGURES / "variance_reduction.png",
                                  "How much each adjustment shrinks the variance")

    out["runtime_sec"] = round(time.time() - t0, 1)
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"Done in {out['runtime_sec']}s")


if __name__ == "__main__":
    main()
