import numpy as np

def _generate_get_feature_names_out(estimator, n_features_out, input_features=None):
    """Generate feature names out for estimator using the estimator name as the prefix.

    The input_feature names are validated but not used. This function is useful
    for estimators that generate their own names based on `n_features_out`, i.e. PCA.

    Parameters
    ----------
    estimator : estimator instance
        Estimator producing output feature names.

    n_feature_out : int
        Number of feature names out.

    input_features : array-like of str or None, default=None
        Only used to validate feature names with `estimator.feature_names_in_`.

    Returns
    -------
    feature_names_in : ndarray of str or `None`
        Feature names in.
    """
    _check_feature_names_in(estimator, input_features, generate_names=False)
    estimator_name = estimator.__class__.__name__.lower()
    return np.asarray(
        [f"{estimator_name}{i}" for i in range(n_features_out)], dtype=object
    )
