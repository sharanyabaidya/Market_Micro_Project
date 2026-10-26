"""Inference helpers. Events from the same stock-day are not independent, so every standard error
is clustered by stock-day (FI-2010) or by stock x 30-minute block (LOBSTER, one day only)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def cluster_mean(x, groups):
    """Mean of x with a cluster-robust standard error (CR1). Returns dict."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(groups)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    if n < 3:
        return dict(mean=np.nan, se=np.nan, t=np.nan, p=np.nan, n=n, clusters=0)
    m = x.mean()
    sums = pd.Series(x - m).groupby(g).sum().to_numpy()
    G = len(sums)
    se = np.sqrt((sums ** 2).sum() * G / (G - 1)) / n if G > 1 else np.nan
    tval = m / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(tval), df=max(G - 1, 1)) if np.isfinite(tval) else np.nan
    return dict(mean=m, se=se, t=tval, p=p, n=n, clusters=G)


def path_ci(A, groups):
    """Per-column cluster mean and 95% CI for a matrix A (events x taus)."""
    res = [cluster_mean(A[:, j], groups) for j in range(A.shape[1])]
    m = np.array([r["mean"] for r in res])
    se = np.array([r["se"] for r in res])
    return m, m - 1.96 * se, m + 1.96 * se


def ols_cluster(data, y, xvars, fe=None, cluster=None):
    """OLS with optional fixed-effect dummies and cluster-robust SEs. Returns fitted results."""
    d = data[[y, *xvars] + ([fe] if fe else []) + ([cluster] if cluster and cluster != fe else [])].dropna()
    X = d[xvars].astype(float)
    if fe:
        X = pd.concat([X, pd.get_dummies(d[fe], prefix="fe", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    kw = {}
    if cluster:
        kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return sm.OLS(d[y].astype(float), X).fit(**kw)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
