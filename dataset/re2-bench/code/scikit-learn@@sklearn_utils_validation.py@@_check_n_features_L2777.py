def _check_n_features(estimator, X, reset):
    """Set the `n_features_in_` attribute, or check against it on an estimator.

    .. note::
        To only check n_features without conducting a full data validation, prefer
        using `validate_data(..., skip_check_array=True)` if possible.

    .. versionchanged:: 1.6
        Moved from :class:`~sklearn.base.BaseEstimator` to
        :mod:`~sklearn.utils.validation`.

    Parameters
    ----------
    estimator : estimator instance
        The estimator to validate the input for.

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    reset : bool
        Whether to reset the `n_features_in_` attribute.
        If True, the `n_features_in_` attribute is set to `X.shape[1]`.
        If False and the attribute exists, then check that it is equal to
        `X.shape[1]`. If False and the attribute does *not* exist, then
        the check is skipped.

        .. note::
           It is recommended to call `reset=True` in `fit` and in the first
           call to `partial_fit`. All other methods that validate `X`
           should set `reset=False`.
    """
    try:
        n_features = _num_features(X)
    except TypeError as e:
        if not reset and hasattr(estimator, "n_features_in_"):
            raise ValueError(
                "X does not contain any features, but "
                f"{estimator.__class__.__name__} is expecting "
                f"{estimator.n_features_in_} features"
            ) from e
        # If the number of features is not defined and reset=True,
        # then we skip this check
        return

    if reset:
        estimator.n_features_in_ = n_features
        return

    if not hasattr(estimator, "n_features_in_"):
        # Skip this check if the expected number of expected input features
        # was not recorded by calling fit first. This is typically the case
        # for stateless transformers.
        return

    if n_features != estimator.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {estimator.__class__.__name__} "
            f"is expecting {estimator.n_features_in_} features as input."
        )
