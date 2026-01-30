import numpy as np
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._mask import _get_mask
from sklearn.utils._missing import is_pandas_na, is_scalar_nan
from sklearn.utils._param_validation import MissingValues, StrOptions
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_feature_names_in,
    _check_n_features,
    check_is_fitted,
    validate_data,
)

class MissingIndicator(TransformerMixin, BaseEstimator):
    """Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`~sklearn.pipeline.Pipeline` consisting of transformers and a
    classifier, but rather could be added using a
    :class:`~sklearn.pipeline.FeatureUnion` or
    :class:`~sklearn.compose.ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    features : {'missing-only', 'all'}, default='missing-only'
        Whether the imputer mask should represent all or a subset of
        features.

        - If `'missing-only'` (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If `'all'`, the imputer mask will represent all features.

    sparse : bool or 'auto', default='auto'
        Whether the imputer mask format should be sparse or dense.

        - If `'auto'` (default), the imputer mask will be of same type as
          input.
        - If `True`, the imputer mask will be a sparse matrix.
        - If `False`, the imputer mask will be a numpy array.

    error_on_new : bool, default=True
        If `True`, :meth:`transform` will raise an error when there are
        features with missing values that have no missing values in
        :meth:`fit`. This is applicable only when `features='missing-only'`.

    Attributes
    ----------
    features_ : ndarray of shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling
        :meth:`transform`. They are computed during :meth:`fit`. If
        `features='all'`, `features_` is equal to `range(n_features)`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SimpleImputer : Univariate imputation of missing values.
    IterativeImputer : Multivariate imputation of missing values.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)
    MissingIndicator()
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])
    """
    _parameter_constraints: dict = {'missing_values': [MissingValues()], 'features': [StrOptions({'missing-only', 'all'})], 'sparse': ['boolean', StrOptions({'auto'})], 'error_on_new': ['boolean']}

    def __init__(self, *, missing_values=np.nan, features='missing-only', sparse='auto', error_on_new=True):
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new

    def _get_missing_features_info(self, X):
        """Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data with missing values. Note that `X` has been
            checked in :meth:`fit` and :meth:`transform` before to call this
            function.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape         (n_samples, n_features)
            The imputer mask of the original data.

        features_with_missing : ndarray of shape (n_features_with_missing)
            The features containing missing values.
        """
        if not self._precomputed:
            imputer_mask = _get_mask(X, self.missing_values)
        else:
            imputer_mask = X
        if sp.issparse(X):
            imputer_mask.eliminate_zeros()
            if self.features == 'missing-only':
                n_missing = imputer_mask.sum(axis=0)
            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == 'csr':
                imputer_mask = imputer_mask.tocsc()
        else:
            if not self._precomputed:
                imputer_mask = _get_mask(X, self.missing_values)
            else:
                imputer_mask = X
            if self.features == 'missing-only':
                n_missing = imputer_mask.sum(axis=0)
            if self.sparse is True:
                imputer_mask = sp.csc_matrix(imputer_mask)
        if self.features == 'all':
            features_indices = np.arange(X.shape[1])
        else:
            features_indices = np.flatnonzero(n_missing)
        return (imputer_mask, features_indices)

    def _validate_input(self, X, in_fit):
        if not is_scalar_nan(self.missing_values):
            ensure_all_finite = True
        else:
            ensure_all_finite = 'allow-nan'
        X = validate_data(self, X, reset=in_fit, accept_sparse=('csc', 'csr'), dtype=None, ensure_all_finite=ensure_all_finite)
        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ('i', 'u', 'f', 'O'):
            raise ValueError('MissingIndicator does not support data with dtype {0}. Please provide either a numeric array (with a floating point or integer dtype) or categorical data represented either as an array with integer dtype or an array of string values with an object dtype.'.format(X.dtype))
        if sp.issparse(X) and self.missing_values == 0:
            raise ValueError('Sparse input with missing_values=0 is not supported. Provide a dense array instead.')
        return X

    def _fit(self, X, y=None, precomputed=False):
        """Fit the transformer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            If `precomputed=True`, then `X` is a mask of the input data.

        precomputed : bool
            Whether the input data is a mask.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape (n_samples,         n_features)
            The imputer mask of the original data.
        """
        if precomputed:
            if not (hasattr(X, 'dtype') and X.dtype.kind == 'b'):
                raise ValueError('precomputed is True but the input data is not a mask')
            self._precomputed = True
        else:
            self._precomputed = False
        if not self._precomputed:
            X = self._validate_input(X, in_fit=True)
        else:
            _check_n_features(self, X, reset=True)
        self._n_features = X.shape[1]
        missing_features_info = self._get_missing_features_info(X)
        self.features_ = missing_features_info[1]
        return missing_features_info[0]
