from typing import Callable
import numpy as np
from sklearn.utils._mask import _get_mask
from sklearn.utils._param_validation import MissingValues, StrOptions
from sklearn.utils.sparsefuncs import _get_median

class SimpleImputer(_BaseImputer):
    """Univariate imputer for completing missing values with simple strategies.

    Replace missing values using a descriptive statistic (e.g. mean, median, or
    most frequent) along each column, or using a constant value.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20
       `SimpleImputer` replaces the previous `sklearn.preprocessing.Imputer`
       estimator which is now removed.

    Parameters
    ----------
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.

    strategy : str or Callable, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
        - If an instance of Callable, then replace missing values using the
          scalar statistic returned by running the callable over a dense 1d
          array containing non-missing values of each column.

        .. versionadded:: 0.20
           strategy="constant" for fixed value imputation.

        .. versionadded:: 1.5
           strategy=callable for custom value imputation.

    fill_value : str or numerical value, default=None
        When strategy == "constant", `fill_value` is used to replace all
        occurrences of missing_values. For string or object data types,
        `fill_value` must be a string.
        If `None`, `fill_value` will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If `X` is not an array of floating values;
        - If `X` is encoded as a CSR matrix;
        - If `add_indicator=True`.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    keep_empty_features : bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0` except when `strategy="constant"`
        in which case `fill_value` will be used instead.

        .. versionadded:: 1.2

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.
        Computing statistics can result in `np.nan` values.
        During :meth:`transform`, features corresponding to `np.nan`
        statistics will be discarded.

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        `None` if `add_indicator=False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    IterativeImputer : Multivariate imputer that estimates values to impute for
        each feature with missing values from all the others.
    KNNImputer : Multivariate imputer that estimates missing features using
        nearest samples.

    Notes
    -----
    Columns which only contained missing values at :meth:`fit` are discarded
    upon :meth:`transform` if strategy is not `"constant"`.

    In a prediction context, simple imputation usually performs poorly when
    associated with a weak learner. However, with a powerful learner, it can
    lead to as good or better performance than complex imputation such as
    :class:`~sklearn.impute.IterativeImputer` or :class:`~sklearn.impute.KNNImputer`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    SimpleImputer()
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

    For a more detailed example see
    :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.
    """
    _parameter_constraints: dict = {**_BaseImputer._parameter_constraints, 'strategy': [StrOptions({'mean', 'median', 'most_frequent', 'constant'}), callable], 'fill_value': 'no_validation', 'copy': ['boolean']}

    def __init__(self, *, missing_values=np.nan, strategy='mean', fill_value=None, copy=True, add_indicator=False, keep_empty_features=False):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator, keep_empty_features=keep_empty_features)
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy

    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on sparse data."""
        missing_mask = _get_mask(X, missing_values)
        mask_data = missing_mask.data
        n_implicit_zeros = X.shape[0] - np.diff(X.indptr)
        statistics = np.empty(X.shape[1])
        if strategy == 'constant':
            statistics.fill(fill_value)
            if not self.keep_empty_features:
                for i in range(missing_mask.shape[1]):
                    if all(missing_mask[:, i].data):
                        statistics[i] = np.nan
        else:
            for i in range(X.shape[1]):
                column = X.data[X.indptr[i]:X.indptr[i + 1]]
                mask_column = mask_data[X.indptr[i]:X.indptr[i + 1]]
                column = column[~mask_column]
                mask_zeros = _get_mask(column, 0)
                column = column[~mask_zeros]
                n_explicit_zeros = mask_zeros.sum()
                n_zeros = n_implicit_zeros[i] + n_explicit_zeros
                if len(column) == 0 and self.keep_empty_features:
                    statistics[i] = 0
                elif strategy == 'mean':
                    s = column.size + n_zeros
                    statistics[i] = np.nan if s == 0 else column.sum() / s
                elif strategy == 'median':
                    statistics[i] = _get_median(column, n_zeros)
                elif strategy == 'most_frequent':
                    statistics[i] = _most_frequent(column, 0, n_zeros)
                elif isinstance(strategy, Callable):
                    statistics[i] = self.strategy(column)
        super()._fit_indicator(missing_mask)
        return statistics
