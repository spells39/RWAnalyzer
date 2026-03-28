from scipy.spatial.distance import jensenshannon
import numpy as np


def _normalize_probabilities(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total == 0:
        raise ValueError("probabilities sum to 0")
    return p / total


def _bootstrap_pmf(p, n, rng):
    idx = rng.choice(len(p), size=n, replace=True, p=p)
    counts = np.bincount(idx, minlength=len(p)).astype(float)
    return counts / counts.sum()


def bootstrap_jsd(prob_real, prob_model, n_bootstrap=1000, sample_size=None, random_state=None):
    """
    Compute a bootstrap p-value for Jensen-Shannon divergence between two PMFs.
    """
    p = _normalize_probabilities(prob_real)
    q = _normalize_probabilities(prob_model)
    if len(p) != len(q):
        raise ValueError("prob_real and prob_model must have the same length")

    n = sample_size or len(p)
    rng = np.random.default_rng(random_state)

    jsd_observed = jensenshannon(p, q)
    jsd_bootstrap = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        p_b = _bootstrap_pmf(p, n, rng)
        q_b = _bootstrap_pmf(q, n, rng)
        jsd_bootstrap[i] = jensenshannon(p_b, q_b)

    return float(np.mean(jsd_bootstrap >= jsd_observed))
