import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils import check_array, metadata_routing

class EmpiricalCovariance(BaseEstimator):
    """Maximum likelihood covariance estimator.

    Read more in the :ref:`User Guide <covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        (stored only if store_precision is True)

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    EllipticEnvelope : An object for detecting outliers in
        a Gaussian distributed dataset.
    GraphicalLasso : Sparse inverse covariance estimation
        with an l1-penalized estimator.
    LedoitWolf : LedoitWolf Estimator.
    MinCovDet : Minimum Covariance Determinant
        (robust estimator of covariance).
    OAS : Oracle Approximating Shrinkage Estimator.
    ShrunkCovariance : Covariance estimator with shrinkage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import EmpiricalCovariance
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> real_cov = np.array([[.8, .3],
    ...                      [.3, .4]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0],
    ...                             cov=real_cov,
    ...                             size=500)
    >>> cov = EmpiricalCovariance().fit(X)
    >>> cov.covariance_
    array([[0.7569, 0.2818],
           [0.2818, 0.3928]])
    >>> cov.location_
    array([0.0622, 0.0193])
    """
    __metadata_request__score = {'X_test': metadata_routing.UNUSED}
    _parameter_constraints: dict = {'store_precision': ['boolean'], 'assume_centered': ['boolean']}

    def __init__(self, *, store_precision=True, assume_centered=False):
        self.store_precision = store_precision
        self.assume_centered = assume_centered

    def error_norm(self, comp_cov, norm='frobenius', scaling=True, squared=True):
        """Compute the Mean Squared Error between two covariance estimators.

        Parameters
        ----------
        comp_cov : array-like of shape (n_features, n_features)
            The covariance to compare with.

        norm : {"frobenius", "spectral"}, default="frobenius"
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            where A is the error ``(comp_cov - self.covariance_)``.

        scaling : bool, default=True
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        squared : bool, default=True
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.

        Returns
        -------
        result : float
            The Mean Squared Error (in the sense of the Frobenius norm) between
            `self` and `comp_cov` covariance estimators.
        """
        error = comp_cov - self.covariance_
        if norm == 'frobenius':
            squared_norm = np.sum(error ** 2)
        elif norm == 'spectral':
            squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
        else:
            raise NotImplementedError('Only spectral and frobenius norms are implemented')
        if scaling:
            squared_norm = squared_norm / error.shape[0]
        if squared:
            result = squared_norm
        else:
            result = np.sqrt(squared_norm)
        return result
