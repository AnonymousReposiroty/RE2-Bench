def _search_estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    Calling a prediction method will only be available if `refit=True`. In
    such case, we check first the fitted best estimator. If it is not
    fitted, we check the unfitted estimator.

    Checking the unfitted estimator allows to use `hasattr` on the `SearchCV`
    instance even before calling `fit`.
    """

    def check(self):
        _check_refit(self, attr)
        if hasattr(self, "best_estimator_"):
            # raise an AttributeError if `attr` does not exist
            getattr(self.best_estimator_, attr)
            return True
        # raise an AttributeError if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check
