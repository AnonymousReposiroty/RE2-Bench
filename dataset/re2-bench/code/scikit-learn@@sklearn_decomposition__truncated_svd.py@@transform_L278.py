from numbers import Integral, Real
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import _randomized_svd, safe_sparse_dot, svd_flip
from sklearn.utils.validation import check_is_fitted, validate_data

class TruncatedSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Dimensionality reduction using truncated SVD (aka LSA).

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition. This means it can work with sparse matrices
    efficiently.

    In particular, truncated SVD works on term count/tf-idf matrices as
    returned by the vectorizers in :mod:`sklearn.feature_extraction.text`. In
    that context, it is known as latent semantic analysis (LSA).

    This estimator supports two algorithms: a fast randomized SVD solver, and
    a "naive" algorithm that uses ARPACK as an eigensolver on `X * X.T` or
    `X.T * X`, whichever is more efficient.

    Read more in the :ref:`User Guide <LSA>`.

    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data.
        If algorithm='arpack', must be strictly less than the number of features.
        If algorithm='randomized', must be less than or equal to the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    algorithm : {'arpack', 'randomized'}, default='randomized'
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    n_iter : int, default=5
        Number of iterations for randomized SVD solver. Not used by ARPACK. The
        default is larger than the default in
        :func:`~sklearn.utils.extmath.randomized_svd` to handle sparse
        matrices that may have large slowly decaying spectrum.

    n_oversamples : int, default=10
        Number of oversamples for randomized SVD solver. Not used by ARPACK.
        See :func:`~sklearn.utils.extmath.randomized_svd` for a complete
        description.

        .. versionadded:: 1.1

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Power iteration normalizer for randomized SVD solver.
        Not used by ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd`
        for more details.

        .. versionadded:: 1.1

    random_state : int, RandomState instance or None, default=None
        Used during randomized svd. Pass an int for reproducible results across
        multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
        SVD solver.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The right singular vectors of the input data.

    explained_variance_ : ndarray of shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    FactorAnalysis : A simple linear generative model with
        Gaussian latent variables.
    IncrementalPCA : Incremental principal components analysis.
    KernelPCA : Kernel Principal component analysis.
    NMF : Non-Negative Matrix Factorization.
    PCA : Principal component analysis.

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.

    References
    ----------
    :arxiv:`Halko, et al. (2009). "Finding structure with randomness:
    Stochastic algorithms for constructing approximate matrix decompositions"
    <0909.4061>`

    Examples
    --------
    >>> from sklearn.decomposition import TruncatedSVD
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_dense = np.random.rand(100, 100)
    >>> X_dense[:, 2 * np.arange(50)] = 0
    >>> X = csr_matrix(X_dense)
    >>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> svd.fit(X)
    TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> print(svd.explained_variance_ratio_)
    [0.0157 0.0512 0.0499 0.0479 0.0453]
    >>> print(svd.explained_variance_ratio_.sum())
    0.2102
    >>> print(svd.singular_values_)
    [35.2410  4.5981   4.5420  4.4486  4.3288]
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'algorithm': [StrOptions({'arpack', 'randomized'})], 'n_iter': [Interval(Integral, 0, None, closed='left')], 'n_oversamples': [Interval(Integral, 1, None, closed='left')], 'power_iteration_normalizer': [StrOptions({'auto', 'OR', 'LU', 'none'})], 'random_state': ['random_state'], 'tol': [Interval(Real, 0, None, closed='left')]}

    def __init__(self, n_components=2, *, algorithm='randomized', n_iter=5, n_oversamples=10, power_iteration_normalizer='auto', random_state=None, tol=0.0):
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.tol = tol

    def transform(self, X):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=['csr', 'csc'], reset=False)
        return safe_sparse_dot(X, self.components_.T)
