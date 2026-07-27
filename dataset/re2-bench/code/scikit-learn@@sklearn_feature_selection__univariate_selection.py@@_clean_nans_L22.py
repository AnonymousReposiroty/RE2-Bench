import numpy as np
from sklearn.utils import as_float_array, check_array, check_X_y, safe_mask, safe_sqr

def _clean_nans(scores):
    """
    Fixes Issue #1240: NaNs can't be properly compared, so change them to the
    smallest value of scores's dtype. -inf seems to be unreliable.
    """
    # XXX where should this function be called? fit? scoring functions
    # themselves?
    scores = as_float_array(scores, copy=True)
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores
