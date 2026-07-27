import warnings
from math import log, sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import _randomized_svd, fast_logdet, squared_norm
from sklearn.utils.validation import check_is_fitted, validate_data

class FactorAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Factor Analysis (FA).

    A simple linear generative model with Gaussian latent variables.

    The observations are assumed to be caused by a linear transformation of
    lower dimensional latent factors and added Gaussian noise.
    Without loss of generality the factors are distributed according to a
    Gaussian with zero mean and unit covariance. The noise is also zero mean
    and has an arbitrary diagonal covariance matrix.

    If we would restrict the model further, by assuming that the Gaussian
    noise is even isotropic (all diagonal entries are the same) we would obtain
    :class:`PCA`.

    FactorAnalysis performs a maximum likelihood estimate of the so-called
    `loading` matrix, the transformation of the latent variables to the
    observed ones, using SVD based approach.

    Read more in the :ref:`User Guide <FA>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int, default=None
        Dimensionality of latent space, the number of components
        of ``X`` that are obtained after ``transform``.
        If None, n_components is set to the number of features.

    tol : float, default=1e-2
        Stopping tolerance for log-likelihood increase.

    copy : bool, default=True
        Whether to make a copy of X. If ``False``, the input X gets overwritten
        during fitting.

    max_iter : int, default=1000
        Maximum number of iterations.

    noise_variance_init : array-like of shape (n_features,), default=None
        The initial guess of the noise variance for each feature.
        If None, it defaults to np.ones(n_features).

    svd_method : {'lapack', 'randomized'}, default='randomized'
        Which SVD method to use. If 'lapack' use standard SVD from
        scipy.linalg, if 'randomized' use fast ``randomized_svd`` function.
        Defaults to 'randomized'. For most applications 'randomized' will
        be sufficiently precise while providing significant speed gains.
        Accuracy can also be improved by setting higher values for
        `iterated_power`. If this is not sufficient, for maximum precision
        you should choose 'lapack'.

    iterated_power : int, default=3
        Number of iterations for the power method. 3 by default. Only used
        if ``svd_method`` equals 'randomized'.

    rotation : {'varimax', 'quartimax'}, default=None
        If not None, apply the indicated rotation. Currently, varimax and
        quartimax are implemented. See
        `"The varimax criterion for analytic rotation in factor analysis"
        <https://link.springer.com/article/10.1007%2FBF02289233>`_
        H. F. Kaiser, 1958.

        .. versionadded:: 0.24

    random_state : int or RandomState instance, default=0
        Only used when ``svd_method`` equals 'randomized'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Components with maximum variance.

    loglike_ : list of shape (n_iterations,)
        The log likelihood at each iteration.

    noise_variance_ : ndarray of shape (n_features,)
        The estimated noise variance for each feature.

    n_iter_ : int
        Number of iterations run.

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PCA: Principal component analysis is also a latent linear variable model
        which however assumes equal noise variance for each feature.
        This extra assumption makes probabilistic PCA faster as it can be
        computed in closed form.
    FastICA: Independent component analysis, a latent variable model with
        non-Gaussian latent variables.

    References
    ----------
    - David Barber, Bayesian Reasoning and Machine Learning,
      Algorithm 21.1.

    - Christopher M. Bishop: Pattern Recognition and Machine Learning,
      Chapter 12.2.4.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import FactorAnalysis
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = FactorAnalysis(n_components=7, random_state=0)
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 0, None, closed='left'), None], 'tol': [Interval(Real, 0.0, None, closed='left')], 'copy': ['boolean'], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'noise_variance_init': ['array-like', None], 'svd_method': [StrOptions({'randomized', 'lapack'})], 'iterated_power': [Interval(Integral, 0, None, closed='left')], 'rotation': [StrOptions({'varimax', 'quartimax'}), None], 'random_state': ['random_state']}

    def __init__(self, n_components=None, *, tol=0.01, copy=True, max_iter=1000, noise_variance_init=None, svd_method='randomized', iterated_power=3, rotation=None, random_state=0):
        self.n_components = n_components
        self.copy = copy
        self.tol = tol
        self.max_iter = max_iter
        self.svd_method = svd_method
        self.noise_variance_init = noise_variance_init
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.rotation = rotation

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the FactorAnalysis model to X using SVD based approach.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Ignored parameter.

        Returns
        -------
        self : object
            FactorAnalysis class instance.
        """
        X = validate_data(self, X, copy=self.copy, dtype=np.float64, force_writeable=True)
        n_samples, n_features = X.shape
        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        nsqrt = sqrt(n_samples)
        llconst = n_features * log(2.0 * np.pi) + n_components
        var = np.var(X, axis=0)
        if self.noise_variance_init is None:
            psi = np.ones(n_features, dtype=X.dtype)
        else:
            if len(self.noise_variance_init) != n_features:
                raise ValueError('noise_variance_init dimension does not with number of features : %d != %d' % (len(self.noise_variance_init), n_features))
            psi = np.array(self.noise_variance_init)
        loglike = []
        old_ll = -np.inf
        SMALL = 1e-12
        if self.svd_method == 'lapack':

            def my_svd(X):
                _, s, Vt = linalg.svd(X, full_matrices=False, check_finite=False)
                return (s[:n_components], Vt[:n_components], squared_norm(s[n_components:]))
        else:
            random_state = check_random_state(self.random_state)

            def my_svd(X):
                _, s, Vt = _randomized_svd(X, n_components, random_state=random_state, n_iter=self.iterated_power)
                return (s, Vt, squared_norm(X) - squared_norm(s))
        for i in range(self.max_iter):
            sqrt_psi = np.sqrt(psi) + SMALL
            s, Vt, unexp_var = my_svd(X / (sqrt_psi * nsqrt))
            s **= 2
            W = np.sqrt(np.maximum(s - 1.0, 0.0))[:, np.newaxis] * Vt
            del Vt
            W *= sqrt_psi
            ll = llconst + np.sum(np.log(s))
            ll += unexp_var + np.sum(np.log(psi))
            ll *= -n_samples / 2.0
            loglike.append(ll)
            if ll - old_ll < self.tol:
                break
            old_ll = ll
            psi = np.maximum(var - np.sum(W ** 2, axis=0), SMALL)
        else:
            warnings.warn('FactorAnalysis did not converge. You might want to increase the number of iterations.', ConvergenceWarning)
        self.components_ = W
        if self.rotation is not None:
            self.components_ = self._rotate(W)
        self.noise_variance_ = psi
        self.loglike_ = loglike
        self.n_iter_ = i + 1
        return self

    def _rotate(self, components, n_components=None, tol=1e-06):
        """Rotate the factor analysis solution."""
        return _ortho_rotation(components.T, method=self.rotation, tol=tol)[:self.n_components]
