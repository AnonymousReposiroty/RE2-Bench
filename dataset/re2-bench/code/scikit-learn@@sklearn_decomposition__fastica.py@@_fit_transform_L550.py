import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import as_float_array, check_array, check_random_state
from sklearn.utils._param_validation import (
    Interval,
    Options,
    StrOptions,
    validate_params,
)
from sklearn.utils.validation import check_is_fitted, validate_data

class FastICA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """FastICA: a fast algorithm for Independent Component Analysis.

    The implementation is based on [1]_.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to use. If None is passed, all are used.

    algorithm : {'parallel', 'deflation'}, default='parallel'
        Specify which algorithm to use for FastICA.

    whiten : str or bool, default='unit-variance'
        Specify the whitening strategy to use.

        - If 'arbitrary-variance', a whitening with variance
          arbitrary is used.
        - If 'unit-variance', the whitening matrix is rescaled to ensure that
          each recovered source has unit variance.
        - If False, the data is already considered to be whitened, and no
          whitening is performed.

        .. versionchanged:: 1.3
            The default value of `whiten` changed to 'unit-variance' in 1.3.

    fun : {'logcosh', 'exp', 'cube'} or callable, default='logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. The derivative should be averaged along its last dimension.
        Example::

            def my_g(x):
                return x ** 3, (3 * x ** 2).mean(axis=-1)

    fun_args : dict, default=None
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}.

    max_iter : int, default=200
        Maximum number of iterations during fit.

    tol : float, default=1e-4
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : array-like of shape (n_components, n_components), default=None
        Initial un-mixing array. If `w_init=None`, then an array of values
        drawn from a normal distribution is used.

    whiten_solver : {"eigh", "svd"}, default="svd"
        The solver to use for whitening.

        - "svd" is more stable numerically if the problem is degenerate, and
          often faster when `n_samples <= n_features`.

        - "eigh" is generally more memory efficient when
          `n_samples >= n_features`, and can be faster when
          `n_samples >= 50 * n_features`.

        .. versionadded:: 1.2

    random_state : int, RandomState instance or None, default=None
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The linear operator to apply to the data to get the independent
        sources. This is equal to the unmixing matrix when ``whiten`` is
        False, and equal to ``np.dot(unmixing_matrix, self.whitening_)`` when
        ``whiten`` is True.

    mixing_ : ndarray of shape (n_features, n_components)
        The pseudo-inverse of ``components_``. It is the linear operator
        that maps independent sources to the data.

    mean_ : ndarray of shape(n_features,)
        The mean over features. Only set if `self.whiten` is True.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge.

    whitening_ : ndarray of shape (n_components, n_features)
        Only set if whiten is 'True'. This is the pre-whitening matrix
        that projects data onto the first `n_components` principal components.

    See Also
    --------
    PCA : Principal component analysis (PCA).
    IncrementalPCA : Incremental principal components analysis (IPCA).
    KernelPCA : Kernel Principal component analysis (KPCA).
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparsePCA : Sparse Principal Components Analysis (SparsePCA).

    References
    ----------
    .. [1] A. Hyvarinen and E. Oja, Independent Component Analysis:
           Algorithms and Applications, Neural Networks, 13(4-5), 2000,
           pp. 411-430.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import FastICA
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = FastICA(n_components=7,
    ...         random_state=0,
    ...         whiten='unit-variance')
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left'), None], 'algorithm': [StrOptions({'parallel', 'deflation'})], 'whiten': [StrOptions({'arbitrary-variance', 'unit-variance'}), Options(bool, {False})], 'fun': [StrOptions({'logcosh', 'exp', 'cube'}), callable], 'fun_args': [dict, None], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'tol': [Interval(Real, 0.0, None, closed='left')], 'w_init': ['array-like', None], 'whiten_solver': [StrOptions({'eigh', 'svd'})], 'random_state': ['random_state']}

    def __init__(self, n_components=None, *, algorithm='parallel', whiten='unit-variance', fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, whiten_solver='svd', random_state=None):
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.whiten_solver = whiten_solver
        self.random_state = random_state

    def _fit_transform(self, X, compute_sources=False):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        compute_sources : bool, default=False
            If False, sources are not computes but only the rotation matrix.
            This can save memory when working with big data. Defaults to False.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components) or None
            Sources matrix. `None` if `compute_sources` is `False`.
        """
        XT = validate_data(self, X, copy=self.whiten, dtype=[np.float64, np.float32], ensure_min_samples=2).T
        fun_args = {} if self.fun_args is None else self.fun_args
        random_state = check_random_state(self.random_state)
        alpha = fun_args.get('alpha', 1.0)
        if not 1 <= alpha <= 2:
            raise ValueError('alpha must be in [1,2]')
        if self.fun == 'logcosh':
            g = _logcosh
        elif self.fun == 'exp':
            g = _exp
        elif self.fun == 'cube':
            g = _cube
        elif callable(self.fun):

            def g(x, fun_args):
                return self.fun(x, **fun_args)
        n_features, n_samples = XT.shape
        n_components = self.n_components
        if not self.whiten and n_components is not None:
            n_components = None
            warnings.warn('Ignoring n_components with whiten=False.')
        if n_components is None:
            n_components = min(n_samples, n_features)
        if n_components > min(n_samples, n_features):
            n_components = min(n_samples, n_features)
            warnings.warn('n_components is too large: it will be set to %s' % n_components)
        if self.whiten:
            X_mean = XT.mean(axis=-1)
            XT -= X_mean[:, np.newaxis]
            if self.whiten_solver == 'eigh':
                d, u = linalg.eigh(XT.dot(X))
                sort_indices = np.argsort(d)[::-1]
                eps = np.finfo(d.dtype).eps * 10
                degenerate_idx = d < eps
                if np.any(degenerate_idx):
                    warnings.warn("There are some small singular values, using whiten_solver = 'svd' might lead to more accurate results.")
                d[degenerate_idx] = eps
                np.sqrt(d, out=d)
                d, u = (d[sort_indices], u[:, sort_indices])
            elif self.whiten_solver == 'svd':
                u, d = linalg.svd(XT, full_matrices=False, check_finite=False)[:2]
            u *= np.sign(u[0])
            K = (u / d).T[:n_components]
            del u, d
            X1 = np.dot(K, XT)
            X1 *= np.sqrt(n_samples)
        else:
            X1 = as_float_array(XT, copy=False)
        w_init = self.w_init
        if w_init is None:
            w_init = np.asarray(random_state.normal(size=(n_components, n_components)), dtype=X1.dtype)
        else:
            w_init = np.asarray(w_init)
            if w_init.shape != (n_components, n_components):
                raise ValueError('w_init has invalid shape -- should be %(shape)s' % {'shape': (n_components, n_components)})
        kwargs = {'tol': self.tol, 'g': g, 'fun_args': fun_args, 'max_iter': self.max_iter, 'w_init': w_init}
        if self.algorithm == 'parallel':
            W, n_iter = _ica_par(X1, **kwargs)
        elif self.algorithm == 'deflation':
            W, n_iter = _ica_def(X1, **kwargs)
        del X1
        self.n_iter_ = n_iter
        if compute_sources:
            if self.whiten:
                S = np.linalg.multi_dot([W, K, XT]).T
            else:
                S = np.dot(W, XT).T
        else:
            S = None
        if self.whiten:
            if self.whiten == 'unit-variance':
                if not compute_sources:
                    S = np.linalg.multi_dot([W, K, XT]).T
                S_std = np.std(S, axis=0, keepdims=True)
                S /= S_std
                W /= S_std.T
            self.components_ = np.dot(W, K)
            self.mean_ = X_mean
            self.whitening_ = K
        else:
            self.components_ = W
        self.mixing_ = linalg.pinv(self.components_, check_finite=False)
        self._unmixing = W
        return S
