import numpy as np
from sklearn.covariance import EmpiricalCovariance, empirical_covariance

def _oas(X, *, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    The formulation is based on [1]_.
    [1] "Shrinkage algorithms for MMSE covariance estimation.",
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
        https://arxiv.org/pdf/0907.4698.pdf
    """
    if len(X.shape) == 2 and X.shape[1] == 1:
        # for only one feature, the result is the same whatever the shrinkage
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X**2).mean()), 0.0

    n_samples, n_features = X.shape

    emp_cov = empirical_covariance(X, assume_centered=assume_centered)

    # The shrinkage is defined as:
    # shrinkage = min(
    # trace(S @ S.T) + trace(S)**2) / ((n + 1) (trace(S @ S.T) - trace(S)**2 / p), 1
    # )
    # where n and p are n_samples and n_features, respectively (cf. Eq. 23 in [1]).
    # The factor 2 / p is omitted since it does not impact the value of the estimator
    # for large p.

    # Instead of computing trace(S)**2, we can compute the average of the squared
    # elements of S that is equal to trace(S)**2 / p**2.
    # See the definition of the Frobenius norm:
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    alpha = np.mean(emp_cov**2)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu**2

    # The factor 1 / p**2 will cancel out since it is in both the numerator and
    # denominator
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)

    # The shrunk covariance is defined as:
    # (1 - shrinkage) * S + shrinkage * F (cf. Eq. 4 in [1])
    # where S is the empirical covariance and F is the shrinkage target defined as
    # F = trace(S) / n_features * np.identity(n_features) (cf. Eq. 3 in [1])
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu

    return shrunk_cov, shrinkage
