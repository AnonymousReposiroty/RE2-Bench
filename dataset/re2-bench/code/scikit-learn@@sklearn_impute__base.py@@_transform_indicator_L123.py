import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import MissingValues, StrOptions

class _BaseImputer(TransformerMixin, BaseEstimator):
    """Base class for all imputers.

    It adds automatically support for `add_indicator`.
    """
    _parameter_constraints: dict = {'missing_values': [MissingValues()], 'add_indicator': ['boolean'], 'keep_empty_features': ['boolean']}

    def __init__(self, *, missing_values=np.nan, add_indicator=False, keep_empty_features=False):
        self.missing_values = missing_values
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _transform_indicator(self, X):
        """Compute the indicator mask.'

        Note that X must be the original data as passed to the imputer before
        any imputation, since imputation may be done inplace in some cases.
        """
        if self.add_indicator:
            if not hasattr(self, 'indicator_'):
                raise ValueError('Make sure to call _fit_indicator before _transform_indicator')
            return self.indicator_.transform(X)
