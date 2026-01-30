from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval, Options, StrOptions
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    check_array,
    check_is_fitted,
    validate_data,
)

class KBinsDiscretizer(TransformerMixin, BaseEstimator):
    """
    Bin continuous data into intervals.

    Read more in the :ref:`User Guide <preprocessing_discretization>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    n_bins : int or array-like of shape (n_features,), default=5
        The number of bins to produce. Raises ValueError if ``n_bins < 2``.

    encode : {'onehot', 'onehot-dense', 'ordinal'}, default='onehot'
        Method used to encode the transformed result.

        - 'onehot': Encode the transformed result with one-hot encoding
          and return a sparse matrix. Ignored features are always
          stacked to the right.
        - 'onehot-dense': Encode the transformed result with one-hot encoding
          and return a dense array. Ignored features are always
          stacked to the right.
        - 'ordinal': Return the bin identifier encoded as an integer value.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.

        - 'uniform': All bins in each feature have identical widths.
        - 'quantile': All bins in each feature have the same number of points.
        - 'kmeans': Values in each bin have the same nearest center of a 1D
          k-means cluster.

        For an example of the different strategies see:
        :ref:`sphx_glr_auto_examples_preprocessing_plot_discretization_strategies.py`.

    quantile_method : {"inverted_cdf", "averaged_inverted_cdf",
            "closest_observation", "interpolated_inverted_cdf", "hazen",
            "weibull", "linear", "median_unbiased", "normal_unbiased"},
            default="linear"
            Method to pass on to np.percentile calculation when using
            strategy="quantile". Only `averaged_inverted_cdf` and `inverted_cdf`
            support the use of `sample_weight != None` when subsampling is not
            active.

            .. versionadded:: 1.7

    dtype : {np.float32, np.float64}, default=None
        The desired data-type for the output. If None, output dtype is
        consistent with input dtype. Only np.float32 and np.float64 are
        supported.

        .. versionadded:: 0.24

    subsample : int or None, default=200_000
        Maximum number of samples, used to fit the model, for computational
        efficiency.
        `subsample=None` means that all the training samples are used when
        computing the quantiles that determine the binning thresholds.
        Since quantile computation relies on sorting each column of `X` and
        that sorting has an `n log(n)` time complexity,
        it is recommended to use subsampling on datasets with a
        very large number of samples.

        .. versionchanged:: 1.3
            The default value of `subsample` changed from `None` to `200_000` when
            `strategy="quantile"`.

        .. versionchanged:: 1.5
            The default value of `subsample` changed from `None` to `200_000` when
            `strategy="uniform"` or `strategy="kmeans"`.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling.
        Pass an int for reproducible results across multiple function calls.
        See the `subsample` parameter for more details.
        See :term:`Glossary <random_state>`.

        .. versionadded:: 1.1

    Attributes
    ----------
    bin_edges_ : ndarray of ndarray of shape (n_features,)
        The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.

    n_bins_ : ndarray of shape (n_features,), dtype=np.int64
        Number of bins per feature. Bins whose width are too small
        (i.e., <= 1e-8) are removed with a warning.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Binarizer : Class used to bin values as ``0`` or
        ``1`` based on a parameter ``threshold``.

    Notes
    -----

    For a visualization of discretization on different datasets refer to
    :ref:`sphx_glr_auto_examples_preprocessing_plot_discretization_classification.py`.
    On the effect of discretization on linear models see:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_discretization.py`.

    In bin edges for feature ``i``, the first and last values are used only for
    ``inverse_transform``. During transform, bin edges are extended to::

      np.concatenate([-np.inf, bin_edges_[i][1:-1], np.inf])

    You can combine ``KBinsDiscretizer`` with
    :class:`~sklearn.compose.ColumnTransformer` if you only want to preprocess
    part of the features.

    ``KBinsDiscretizer`` might produce constant features (e.g., when
    ``encode = 'onehot'`` and certain bins do not contain any data).
    These features can be removed with feature selection algorithms
    (e.g., :class:`~sklearn.feature_selection.VarianceThreshold`).

    Examples
    --------
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> X = [[-2, 1, -4,   -1],
    ...      [-1, 2, -3, -0.5],
    ...      [ 0, 3, -2,  0.5],
    ...      [ 1, 4, -1,    2]]
    >>> est = KBinsDiscretizer(
    ...     n_bins=3, encode='ordinal', strategy='uniform'
    ... )
    >>> est.fit(X)
    KBinsDiscretizer(...)
    >>> Xt = est.transform(X)
    >>> Xt  # doctest: +SKIP
    array([[ 0., 0., 0., 0.],
           [ 1., 1., 1., 0.],
           [ 2., 2., 2., 1.],
           [ 2., 2., 2., 2.]])

    Sometimes it may be useful to convert the data back into the original
    feature space. The ``inverse_transform`` function converts the binned
    data into the original feature space. Each value will be equal to the mean
    of the two bin edges.

    >>> est.bin_edges_[0]
    array([-2., -1.,  0.,  1.])
    >>> est.inverse_transform(Xt)
    array([[-1.5,  1.5, -3.5, -0.5],
           [-0.5,  2.5, -2.5, -0.5],
           [ 0.5,  3.5, -1.5,  0.5],
           [ 0.5,  3.5, -1.5,  1.5]])

    While this preprocessing step can be an optimization, it is important
    to note the array returned by ``inverse_transform`` will have an internal type
    of ``np.float64`` or ``np.float32``, denoted by the ``dtype`` input argument.
    This can drastically increase the memory usage of the array. See the
    :ref:`sphx_glr_auto_examples_cluster_plot_face_compress.py`
    where `KBinsDescretizer` is used to cluster the image into bins and increases
    the size of the image by 8x.
    """
    _parameter_constraints: dict = {'n_bins': [Interval(Integral, 2, None, closed='left'), 'array-like'], 'encode': [StrOptions({'onehot', 'onehot-dense', 'ordinal'})], 'strategy': [StrOptions({'uniform', 'quantile', 'kmeans'})], 'quantile_method': [StrOptions({'warn', 'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased', 'normal_unbiased'})], 'dtype': [Options(type, {np.float64, np.float32}), None], 'subsample': [Interval(Integral, 1, None, closed='left'), None], 'random_state': ['random_state']}

    def __init__(self, n_bins=5, *, encode='onehot', strategy='quantile', quantile_method='warn', dtype=None, subsample=200000, random_state=None):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state

    def _validate_n_bins(self, n_features):
        """Returns n_bins_, the number of bins per feature."""
        orig_bins = self.n_bins
        if isinstance(orig_bins, Integral):
            return np.full(n_features, orig_bins, dtype=int)
        n_bins = check_array(orig_bins, dtype=int, copy=True, ensure_2d=False)
        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError('n_bins must be a scalar or array of shape (n_features,).')
        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)
        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ', '.join((str(i) for i in violating_indices))
            raise ValueError('{} received an invalid number of bins at indices {}. Number of bins must be at least 2, and must be an int.'.format(KBinsDiscretizer.__name__, indices))
        return n_bins
