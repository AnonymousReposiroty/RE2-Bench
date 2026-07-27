import numpy as np
from scipy import sparse, stats
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._array_api import (
    _find_matching_floating_dtype,
    _max_precision_float_dtype,
    _modify_in_place_if_numpy,
    device,
    get_namespace,
    get_namespace_and_device,
    size,
    supported_float_dtypes,
)
from sklearn.utils.extmath import _incremental_mean_and_var, row_norms
from sklearn.utils.sparsefuncs import (
    incr_mean_variance_axis,
    inplace_column_scale,
    mean_variance_axis,
    min_max_axis,
)
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_sample_weight,
    check_is_fitted,
    check_random_state,
    validate_data,
)

class StandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

    .. code-block:: text

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    than others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    `StandardScaler` is sensitive to outliers, and the features may scale
    differently from each other in the presence of outliers. For an example
    visualization, refer to :ref:`Compare StandardScaler with other scalers
    <plot_all_scaling_standard_scaler_section>`.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Generally this is calculated using `np.sqrt(var_)`. If a
        variance is zero, we can't achieve unit variance, and the data is left
        as-is, giving a scaling factor of 1. `scale_` is equal to `None`
        when `with_std=False`.

        .. versionadded:: 0.17
           *scale_*

    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False`` and ``with_std=False``.

    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_mean=False`` and
        ``with_std=False``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_seen_ : int or ndarray of shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are no missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array of dtype int. If
        `sample_weights` are used it will be a float (if no missing data)
        or an array of dtype float that sums the weights seen so far.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    See Also
    --------
    scale : Equivalent function without the estimator API.

    :class:`~sklearn.decomposition.PCA` : Further removes the linear
        correlation across features with 'whiten=True'.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler()
    >>> print(scaler.mean_)
    [0.5 0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[3. 3.]]
    """
    _parameter_constraints: dict = {'copy': ['boolean'], 'with_mean': ['boolean'], 'with_std': ['boolean']}

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, sample_weight=None):
        """Online computation of mean and std on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        xp, _, X_device = get_namespace_and_device(X)
        first_call = not hasattr(self, 'n_samples_seen_')
        X = validate_data(self, X, accept_sparse=('csr', 'csc'), dtype=supported_float_dtypes(xp, X_device), ensure_all_finite='allow-nan', reset=first_call)
        n_features = X.shape[1]
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        dtype = xp.int64 if sample_weight is None else X.dtype
        if first_call:
            self.n_samples_seen_ = xp.zeros(n_features, dtype=dtype, device=X_device)
        elif size(self.n_samples_seen_) == 1:
            self.n_samples_seen_ = xp.repeat(self.n_samples_seen_, X.shape[1])
            self.n_samples_seen_ = xp.astype(self.n_samples_seen_, dtype, copy=False)
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError('Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.')
            sparse_constructor = sparse.csr_matrix if X.format == 'csr' else sparse.csc_matrix
            if self.with_std:
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_, self.n_samples_seen_ = mean_variance_axis(X, axis=0, weights=sample_weight, return_sum_weights=True)
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = incr_mean_variance_axis(X, axis=0, last_mean=self.mean_, last_var=self.var_, last_n=self.n_samples_seen_, weights=sample_weight)
                self.mean_ = self.mean_.astype(np.float64, copy=False)
                self.var_ = self.var_.astype(np.float64, copy=False)
            else:
                self.mean_ = None
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor((np.isnan(X.data), X.indices, X.indptr), shape=X.shape)
                self.n_samples_seen_ += (np.sum(weights) - sum_weights_nan).astype(dtype)
        else:
            if not hasattr(self, 'scale_'):
                self.mean_ = 0.0
                if self.with_std:
                    self.var_ = 0.0
                else:
                    self.var_ = None
            if not self.with_mean and (not self.with_std):
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - xp.isnan(X).sum(axis=0)
            else:
                self.mean_, self.var_, self.n_samples_seen_ = _incremental_mean_and_var(X, self.mean_, self.var_, self.n_samples_seen_, sample_weight=sample_weight)
        if xp.max(self.n_samples_seen_) == xp.min(self.n_samples_seen_):
            self.n_samples_seen_ = self.n_samples_seen_[0]
        if self.with_std:
            constant_mask = _is_constant_feature(self.var_, self.mean_, self.n_samples_seen_)
            self.scale_ = _handle_zeros_in_scale(xp.sqrt(self.var_), copy=False, constant_mask=constant_mask)
        else:
            self.scale_ = None
        return self
