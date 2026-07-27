from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import (
    _allclose_dense_sparse,
    _check_feature_names_in,
    _get_feature_names,
    _is_pandas_df,
    _is_polars_df,
    check_array,
    validate_data,
)

class FunctionTransformer(TransformerMixin, BaseEstimator):
    """Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <function_transformer>`.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.

    inverse_func : callable, default=None
        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If inverse_func is None, then inverse_func
        will be the identity function.

    validate : bool, default=False
        Indicate that the input X array should be checked before calling
        ``func``. The possibilities are:

        - If False, there is no input validation.
        - If True, then X will be converted to a 2-dimensional NumPy array or
          sparse matrix. If the conversion is not possible an exception is
          raised.

        .. versionchanged:: 0.22
           The default of ``validate`` changed from True to False.

    accept_sparse : bool, default=False
        Indicate that func accepts a sparse matrix as input. If validate is
        False, this has no effect. Otherwise, if accept_sparse is false,
        sparse matrix inputs will cause an exception to be raised.

    check_inverse : bool, default=True
       Whether to check that or ``func`` followed by ``inverse_func`` leads to
       the original inputs. It can be used for a sanity check, raising a
       warning when the condition is not fulfilled.

       .. versionadded:: 0.20

    feature_names_out : callable, 'one-to-one' or None, default=None
        Determines the list of feature names that will be returned by the
        `get_feature_names_out` method. If it is 'one-to-one', then the output
        feature names will be equal to the input feature names. If it is a
        callable, then it must take two positional arguments: this
        `FunctionTransformer` (`self`) and an array-like of input feature names
        (`input_features`). It must return an array-like of output feature
        names. The `get_feature_names_out` method is only defined if
        `feature_names_out` is not None.

        See ``get_feature_names_out`` for more details.

        .. versionadded:: 1.1

    kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to func.

        .. versionadded:: 0.18

    inv_kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to inverse_func.

        .. versionadded:: 0.18

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MaxAbsScaler : Scale each feature by its maximum absolute value.
    StandardScaler : Standardize features by removing the mean and
        scaling to unit variance.
    LabelBinarizer : Binarize labels in a one-vs-all fashion.
    MultiLabelBinarizer : Transform between iterable of iterables
        and a multilabel format.

    Notes
    -----
    If `func` returns an output with a `columns` attribute, then the columns is enforced
    to be consistent with the output of `get_feature_names_out`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> transformer = FunctionTransformer(np.log1p)
    >>> X = np.array([[0, 1], [2, 3]])
    >>> transformer.transform(X)
    array([[0.       , 0.6931],
           [1.0986, 1.3862]])
    """
    _parameter_constraints: dict = {'func': [callable, None], 'inverse_func': [callable, None], 'validate': ['boolean'], 'accept_sparse': ['boolean'], 'check_inverse': ['boolean'], 'feature_names_out': [callable, StrOptions({'one-to-one'}), None], 'kw_args': [dict, None], 'inv_kw_args': [dict, None]}

    def __init__(self, func=None, inverse_func=None, *, validate=False, accept_sparse=False, check_inverse=True, feature_names_out=None, kw_args=None, inv_kw_args=None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

    def inverse_transform(self, X):
        """Transform X using the inverse function.

        Parameters
        ----------
        X : {array-like, sparse-matrix} of shape (n_samples, n_features)                 if `validate=True` else any object that `inverse_func` can handle
            Input array.

        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if self.validate:
            X = check_array(X, accept_sparse=self.accept_sparse)
        return self._transform(X, func=self.inverse_func, kw_args=self.inv_kw_args)

    def _transform(self, X, func=None, kw_args=None):
        if func is None:
            func = _identity
        return func(X, **kw_args if kw_args else {})
