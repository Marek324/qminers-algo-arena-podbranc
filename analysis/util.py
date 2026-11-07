"""Small statistics helpers whose outputs always have the same keys, whatever the data."""

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats as sps

US = 1_000_000
DIST_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
_Q_KEYS = tuple(f'p{round(q * 100):02d}' for q in DIST_QUANTILES)


def label_s(seconds: float) -> str:
    """'5ms', '100ms', '1s', '300s': stable keys for horizons."""
    if seconds < 1:
        return f'{round(seconds * 1000)}ms'
    return f'{round(seconds)}s'


def _weighted_quantiles(x: np.ndarray, w: np.ndarray, qs: Sequence[float]) -> list[float]:
    order = np.argsort(x, kind='stable')
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    total = cw[-1]
    return [float(xs[min(np.searchsorted(cw, q * total, side='left'), len(xs) - 1)]) for q in qs]


def dist(x: np.ndarray | Sequence[float], w: np.ndarray | None = None) -> dict:
    """Distribution summary: n, mean, std, min, p01..p99, max. Optional weights (e.g. time)."""
    x = np.asarray(x, dtype=float)
    if w is not None:
        w = np.asarray(w, dtype=float)
        ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
        x, w = x[ok], w[ok]
    else:
        x = x[np.isfinite(x)]
    out: dict = {'n': int(len(x))}
    if len(x) == 0:
        out.update({'mean': None, 'std': None, 'min': None})
        out.update({k: None for k in _Q_KEYS})
        out['max'] = None
        return out
    if w is None:
        mean = float(x.mean())
        std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        qv = np.quantile(x, DIST_QUANTILES).tolist()
    else:
        mean = float(np.average(x, weights=w))
        std = float(math.sqrt(np.average((x - mean) ** 2, weights=w)))
        qv = _weighted_quantiles(x, w, DIST_QUANTILES)
    out.update({'mean': mean, 'std': std, 'min': float(x.min())})
    out.update(dict(zip(_Q_KEYS, qv, strict=True)))
    out['max'] = float(x.max())
    return out


def moments(x: np.ndarray) -> dict:
    """Mean, std, skew, excess kurtosis and n of the finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return {'n': int(len(x)), 'mean': None, 'std': None, 'skew': None, 'kurtosis': None}
    std = float(x.std(ddof=1))
    return {
        'n': int(len(x)),
        'mean': float(x.mean()),
        'std': std,
        'skew': float(sps.skew(x)) if std > 0 else None,
        'kurtosis': float(sps.kurtosis(x)) if std > 0 else None,
    }


def categorical_shares(values: np.ndarray, labels: Sequence, weights: np.ndarray | None = None) -> dict:
    """{label: share} for each fixed label (0 when absent). Values outside the labels are ignored."""
    values = np.asarray(values)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    total = float(w.sum())
    out = {}
    for lab in labels:
        out[str(lab)] = float(w[values == lab].sum() / total) if total > 0 else None
    return out


def binned_shares(x: np.ndarray, edges: Sequence[float], names: Sequence[str], weights: np.ndarray | None = None) -> dict:
    """Share of (weighted) x in each [edges[i], edges[i+1]) bin, keyed by names."""
    x = np.asarray(x, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(x)
    x, w = x[ok], w[ok]
    total = float(w.sum())
    out = {}
    for name, lo, hi in zip(names, edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x < hi)
        out[name] = float(w[m].sum() / total) if total > 0 else None
    return out


def corr(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson and Spearman correlation on the pairs where both are finite."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 or x.std() == 0 or y.std() == 0:
        return {'n': int(len(x)), 'pearson': None, 'spearman': None}
    return {
        'n': int(len(x)),
        'pearson': float(np.corrcoef(x, y)[0, 1]),
        'spearman': float(sps.spearmanr(x, y).statistic),
    }


def autocorr(x: np.ndarray, lags: Sequence[int]) -> dict:
    """{f'lag{k}': autocorrelation} of the finite series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {}
    if len(x) > 2:
        x = x - x.mean()
        var = float(np.dot(x, x))
    else:
        var = 0.0
    for k in lags:
        if var <= 0 or len(x) <= k + 2:
            out[f'lag{k}'] = None
        else:
            out[f'lag{k}'] = float(np.dot(x[:-k], x[k:]) / var)
    return out


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """OLS with intercept. Returns coefficients, t-stats and R^2 on the finite rows."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    k = X.shape[1]
    empty = {'n': int(len(y)), 'intercept': None, 'coef': [None] * k, 't': [None] * k, 'r2': None}
    if len(y) < k + 10:
        return empty
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid**2).sum()) / sst if sst > 0 else None
    dof = len(y) - A.shape[1]
    sigma2 = float((resid**2).sum()) / dof
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        t = np.where(se > 0, beta / np.where(se > 0, se, 1), np.nan)
    except np.linalg.LinAlgError:
        t = np.full(len(beta), np.nan)
    return {
        'n': int(len(y)),
        'intercept': float(beta[0]),
        'coef': [float(b) for b in beta[1:]],
        't': [float(v) for v in t[1:]],
        'r2': r2,
    }


def safe_div(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def clean(obj):
    """Recursively turn numpy scalars/arrays into JSON types and NaN/inf into None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            return None
        return float(f'{f:.8g}')
    return obj
