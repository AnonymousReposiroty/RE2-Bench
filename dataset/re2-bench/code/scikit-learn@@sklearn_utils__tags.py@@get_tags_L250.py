def get_tags(estimator) -> Tags:
    """Get estimator tags.

    :class:`~sklearn.BaseEstimator` provides the estimator tags machinery.

    For scikit-learn built-in estimators, we should still rely on
    `self.__sklearn_tags__()`. `get_tags(est)` should be used when we
    are not sure where `est` comes from: typically
    `get_tags(self.estimator)` where `self` is a meta-estimator, or in
    the common checks.

    .. versionadded:: 1.6

    Parameters
    ----------
    estimator : estimator object
        The estimator from which to get the tag.

    Returns
    -------
    tags : :class:`~.sklearn.utils.Tags`
        The estimator tags.
    """

    try:
        tags = estimator.__sklearn_tags__()
    except AttributeError as exc:
        if "object has no attribute '__sklearn_tags__'" in str(exc):
            # Happens when `__sklearn_tags__` is implemented by calling
            # `super().__sklearn_tags__()` but there is no `__sklearn_tags__`
            # method in the base class. Typically happens when only inheriting
            # from Mixins.

            raise AttributeError(
                f"The following error was raised: {exc}. It seems that "
                "there are no classes that implement `__sklearn_tags__` "
                "in the MRO and/or all classes in the MRO call "
                "`super().__sklearn_tags__()`. Make sure to inherit from "
                "`BaseEstimator` which implements `__sklearn_tags__` (or "
                "alternatively define `__sklearn_tags__` but we don't recommend "
                "this approach). Note that `BaseEstimator` needs to be on the "
                "right side of other Mixins in the inheritance order."
            )
        else:
            raise

    return tags
