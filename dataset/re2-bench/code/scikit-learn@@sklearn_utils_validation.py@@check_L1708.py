from collections.abc import Sequence

def _estimator_has(attr, *, delegates=("estimator_", "estimator")):
    """Check if we can delegate a method to the underlying estimator.

    We check the `delegates` in the order they are passed. By default, we first check
    the fitted estimator if available, otherwise we check the unfitted estimator.

    Parameters
    ----------
    attr : str
        Name of the attribute the delegate might or might not have.

    delegates: tuple of str, default=("estimator_", "estimator")
        A tuple of sub-estimator(s) to check if we can delegate the `attr` method.

    Returns
    -------
    check : function
        Function to check if the delegate has the attribute.

    Raises
    ------
    ValueError
        Raised when none of the delegates are present in the object.
    """

    def check(self):
        for delegate in delegates:
            # In meta estimators with multiple sub estimators,
            # only the attribute of the first sub estimator is checked,
            # assuming uniformity across all sub estimators.
            if hasattr(self, delegate):
                delegator = getattr(self, delegate)
                if isinstance(delegator, Sequence):
                    return getattr(delegator[0], attr)
                else:
                    return getattr(delegator, attr)

        raise ValueError(f"None of the delegates {delegates} are present in the class.")

    return check
