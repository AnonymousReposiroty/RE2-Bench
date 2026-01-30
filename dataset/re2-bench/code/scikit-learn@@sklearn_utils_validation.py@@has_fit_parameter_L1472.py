from inspect import Parameter, isclass, signature

def has_fit_parameter(estimator, parameter):
    """Check whether the estimator's fit method supports the given parameter.

    Parameters
    ----------
    estimator : object
        An estimator to inspect.

    parameter : str
        The searched parameter.

    Returns
    -------
    is_parameter : bool
        Whether the parameter was found to be a named parameter of the
        estimator's fit method.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.utils.validation import has_fit_parameter
    >>> has_fit_parameter(SVC(), "sample_weight")
    True
    """
    return (
        # This is used during test collection in common tests. The
        # hasattr(estimator, "fit") makes it so that we don't fail for an estimator
        # that does not have a `fit` method during collection of checks. The right
        # checks will fail later.
        hasattr(estimator, "fit") and parameter in signature(estimator.fit).parameters
    )
