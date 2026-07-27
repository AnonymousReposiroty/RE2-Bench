import numpy as np
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import _safe_indexing, check_array

class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """

    def _check_X(self, X, ensure_all_finite=True):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, e.g. for the `categories_` attribute.

        """
        if not (hasattr(X, 'iloc') and getattr(X, 'ndim', 0) == 2):
            X_temp = check_array(X, dtype=None, ensure_all_finite=ensure_all_finite)
            if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, ensure_all_finite=ensure_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            needs_validation = ensure_all_finite
        n_samples, n_features = X.shape
        X_columns = []
        for i in range(n_features):
            Xi = _safe_indexing(X, indices=i, axis=1)
            Xi = check_array(Xi, ensure_2d=False, dtype=None, ensure_all_finite=needs_validation)
            X_columns.append(Xi)
        return (X_columns, n_samples, n_features)
