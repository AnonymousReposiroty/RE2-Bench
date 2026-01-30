from functools import partial
import numpy as np
from scipy import sparse as sp
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

    def _concatenate_indicator(self, X_imputed, X_indicator):
        """Concatenate indicator mask with the imputed data."""
        if not self.add_indicator:
            return X_imputed
        if sp.issparse(X_imputed):
            hstack = partial(sp.hstack, format=X_imputed.format)
        else:
            hstack = np.hstack
        if X_indicator is None:
            raise ValueError('Data from the missing indicator are not provided. Call _fit_indicator and _transform_indicator in the imputer implementation.')
        return hstack((X_imputed, X_indicator))
