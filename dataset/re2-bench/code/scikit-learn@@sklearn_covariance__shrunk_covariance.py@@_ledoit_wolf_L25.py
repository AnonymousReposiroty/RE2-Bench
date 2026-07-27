import numpy as np
from sklearn.covariance import EmpiricalCovariance, empirical_covariance

def _ledoit_wolf(X, *, assume_centered, block_size):
    """Estimate the shrunk Ledoit-Wolf covariance matrix."""
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X**2).mean()), 0.0
    n_features = X.shape[1]

    # get Ledoit-Wolf shrinkage
    shrinkage = ledoit_wolf_shrinkage(
        X, assume_centered=assume_centered, block_size=block_size
    )
    emp_cov = empirical_covariance(X, assume_centered=assume_centered)
    mu = np.sum(np.trace(emp_cov)) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu

    return shrunk_cov, shrinkage
