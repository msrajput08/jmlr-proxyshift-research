"""Finite-sample conservative score-quantile certificates under proxy-mediated group shift.

Assumptions: known source group probabilities, known stable nonsingular proxy
confusion matrix; proxy and score conditionally independent given latent group;
within-latent-group score distribution invariant between source and target.
"""
import numpy as np


def confusion(error):
    """Symmetric two-group proxy channel, rows=proxy, columns=latent group."""
    return np.array([[1 - error, error], [error, 1 - error]], dtype=float)


def certificate(scores, proxy, source_prior, channel, target_prior,
                alpha=.10, delta=.10, certified=True):
    """Return a threshold with simultaneous calibration-conditional coverage.

    For DKW bounds conditional on proxy stratum counts n_z:
    eps_z = sqrt(log(2*K/delta)/(2*n_z)), with eps=1 for empty stratum.
    The uncertainty bound is sum |b_z| p_z eps_z where
    b = target_prior @ inverse(channel @ diag(source_prior))
    and p = channel @ source_prior.

    If no empirical threshold passes, returns +infinity; this is a valid but
    uninformative prediction region and MUST be reported.
    """
    scores = np.asarray(scores, dtype=float)
    proxy = np.asarray(proxy, dtype=int)
    source_prior = np.asarray(source_prior, dtype=float)
    target_prior = np.asarray(target_prior, dtype=float)
    channel = np.asarray(channel, dtype=float)
    if scores.ndim != 1 or proxy.shape != scores.shape or len(scores)==0:
        raise ValueError('Scores and proxy must be aligned nonempty one-dimensional arrays')
    if not np.isfinite(scores).all() or np.any(scores<0):
        raise ValueError('Nonconformity scores must be finite and nonnegative')
    if not (0<alpha<1 and 0<delta<1):
        raise ValueError('alpha and delta must lie strictly between zero and one')
    k = len(source_prior)
    if channel.shape != (k,k) or target_prior.shape != (k,):
        raise ValueError('Channel and target prior must match source prior dimensions')
    if not np.isfinite(source_prior).all() or not np.isfinite(target_prior).all() or not np.isfinite(channel).all():
        raise ValueError('Priors and channel must be finite')
    if np.any(source_prior<=0) or np.any(target_prior<0) or not np.isclose(source_prior.sum(),1) or not np.isclose(target_prior.sum(),1):
        raise ValueError('Priors must be valid probability vectors with positive source groups')
    if np.any(channel<0) or not np.allclose(channel.sum(axis=0),1):
        raise ValueError('Channel must be nonnegative with stochastic columns')
    if np.any((proxy<0)|(proxy>=k)):
        raise ValueError('Observed proxy indices must be in range')
    a = channel @ np.diag(source_prior)
    b = np.linalg.solve(a.T, target_prior)
    p = channel @ source_prior
    if np.any(p <= 0):
        raise ValueError('All proxy-stratum probabilities must be positive')
    grid = np.sort(scores)
    hat = np.zeros(len(grid))
    penalty = 0.0
    for z in range(k):
        arr = np.sort(scores[proxy == z])
        nz = len(arr)
        eps = min(1.0, np.sqrt(np.log(2*k/delta)/(2*nz))) if nz else 1.0
        penalty += abs(b[z]) * p[z] * eps
        if nz:
            hat += (b[z] * p[z]/nz) * np.searchsorted(arr, grid, side='right')
    lower = hat - (penalty if certified else 0.0)
    idx = np.flatnonzero(lower >= 1.0 - alpha)
    q = float(grid[idx[0]]) if len(idx) else float('inf')
    return q, penalty


def pooled(scores, alpha=.10):
    scores = np.sort(np.asarray(scores))
    idx = min(len(scores)-1, int(np.ceil((len(scores)+1)*(1-alpha)))-1)
    return float(scores[idx])


def latent_oracle(scores, group, source_prior, target_prior, alpha=.10):
    """Diagnostic benchmark using unobserved true calibration labels."""
    grid = np.sort(scores)
    cdf = np.zeros(len(grid))
    for g in range(len(source_prior)):
        vals = np.sort(scores[group == g])
        if len(vals):
            cdf += target_prior[g] * np.searchsorted(vals, grid, side='right')/len(vals)
    idx = np.flatnonzero(cdf >= 1-alpha)
    return float(grid[idx[0]]) if len(idx) else float('inf')


def generate(n, prior_b, channel, rng, scale=(1.0,3.0)):
    group = (rng.random(n)<prior_b).astype(int)
    # Two-group channel: P(proxy=1|G=0)=M[1,0], P(proxy=1|G=1)=M[1,1].
    p1 = channel[1, group]
    proxy = (rng.random(n) < p1).astype(int)
    scores = np.abs(rng.standard_normal(n) * np.take(scale, group))
    return scores, proxy, group
