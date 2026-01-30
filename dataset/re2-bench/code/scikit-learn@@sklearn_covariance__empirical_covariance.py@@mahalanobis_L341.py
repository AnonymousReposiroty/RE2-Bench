import numpy as np
from scipy import linalg
from sklearn import config_context
from sklearn.base import BaseEstimator, _fit_context
from sklearn.metrics.pairwise import pairwise_distances
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

    def mahalanobis(self, X):
        """Compute the squared Mahalanobis distances of given observations.

        For a detailed example of how outliers affects the Mahalanobis distance,
        see :ref:`sphx_glr_auto_examples_covariance_plot_mahalanobis_distances.py`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The observations, the Mahalanobis distances of the which we
            compute. Observations are assumed to be drawn from the same
            distribution than the data used in fit.

        Returns
        -------
        dist : ndarray of shape (n_samples,)
            Squared Mahalanobis distances of the observations.
        """
        X = validate_data(self, X, reset=False)
        precision = self.get_precision()
        with config_context(assume_finite=True):
            dist = pairwise_distances(X, self.location_[np.newaxis, :], metric='mahalanobis', VI=precision)
        return np.reshape(dist, (len(X),)) ** 2
