import scipy.sparse as sp
from sklearn.utils import _safe_indexing

def _check_method_params(X, params, indices=None):
    """Check and validate the parameters passed to a specific
    method like `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    params : dict
        Dictionary containing the parameters passed to the method.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    Returns
    -------
    method_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """
    from sklearn.utils import _safe_indexing

    method_params_validated = {}
    for param_key, param_value in params.items():
        if (
            not _is_arraylike(param_value) and not sp.issparse(param_value)
        ) or _num_samples(param_value) != _num_samples(X):
            # Non-indexable pass-through (for now for backward-compatibility).
            # https://github.com/scikit-learn/scikit-learn/issues/15805
            method_params_validated[param_key] = param_value
        else:
            # Any other method_params should support indexing
            # (e.g. for cross-validation).
            method_params_validated[param_key] = _make_indexable(param_value)
            method_params_validated[param_key] = _safe_indexing(
                method_params_validated[param_key], indices
            )

    return method_params_validated
