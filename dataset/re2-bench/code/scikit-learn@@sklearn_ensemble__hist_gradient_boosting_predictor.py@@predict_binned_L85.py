import numpy as np
from sklearn.ensemble._hist_gradient_boosting._predictor import (
    _compute_partial_dependence,
    _predict_from_binned_data,
    _predict_from_raw_data,
)
from sklearn.ensemble._hist_gradient_boosting.common import (
    PREDICTOR_RECORD_DTYPE,
    Y_DTYPE,
)

class TreePredictor:
    """Tree class used for predictions.

    Parameters
    ----------
    nodes : ndarray of PREDICTOR_RECORD_DTYPE
        The nodes of the tree.
    binned_left_cat_bitsets : ndarray of shape (n_categorical_splits, 8), dtype=uint32
        Array of bitsets for binned categories used in predict_binned when a
        split is categorical.
    raw_left_cat_bitsets : ndarray of shape (n_categorical_splits, 8), dtype=uint32
        Array of bitsets for raw categories used in predict when a split is
        categorical.
    """

    def __init__(self, nodes, binned_left_cat_bitsets, raw_left_cat_bitsets):
        self.nodes = nodes
        self.binned_left_cat_bitsets = binned_left_cat_bitsets
        self.raw_left_cat_bitsets = raw_left_cat_bitsets

    def predict_binned(self, X, missing_values_bin_idx, n_threads):
        """Predict raw values for binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.
        missing_values_bin_idx : uint8
            Index of the bin that is used for missing values. This is the
            index of the last bin and is always equal to max_bins (as passed
            to the GBDT classes), or equivalently to n_bins - 1.
        n_threads : int
            Number of OpenMP threads to use.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
        out = np.empty(X.shape[0], dtype=Y_DTYPE)
        _predict_from_binned_data(self.nodes, X, self.binned_left_cat_bitsets, missing_values_bin_idx, n_threads, out)
        return out
