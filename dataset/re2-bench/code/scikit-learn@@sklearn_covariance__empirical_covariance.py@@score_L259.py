from scipy import linalg
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils import check_array, metadata_routing
from sklearn.utils.validation import validate_data

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

    def get_precision(self):
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : array-like of shape (n_features, n_features)
            The precision matrix associated to the current covariance object.
        """
        if self.store_precision:
            precision = self.precision_
        else:
            precision = linalg.pinvh(self.covariance_, check_finite=False)
        return precision

    def score(self, X_test, y=None):
        """Compute the log-likelihood of `X_test` under the estimated Gaussian model.

        The Gaussian model is defined by its mean and covariance matrix which are
        represented respectively by `self.location_` and `self.covariance_`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data of which we compute the likelihood, where `n_samples` is
            the number of samples and `n_features` is the number of features.
            `X_test` is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        res : float
            The log-likelihood of `X_test` with `self.location_` and `self.covariance_`
            as estimators of the Gaussian model mean and covariance matrix respectively.
        """
        X_test = validate_data(self, X_test, reset=False)
        test_cov = empirical_covariance(X_test - self.location_, assume_centered=True)
        res = log_likelihood(test_cov, self.get_precision())
        return res
