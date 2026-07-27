from math import log
from sklearn.utils._array_api import (
    _max_precision_float_dtype,
    get_namespace_and_device,
)

def _entropy(labels):
    """Calculate the entropy for a labeling.

    Parameters
    ----------
    labels : array-like of shape (n_samples,), dtype=int
        The labels.

    Returns
    -------
    entropy : float
       The entropy for a labeling.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    xp, is_array_api_compliant, device_ = get_namespace_and_device(labels)
    labels_len = labels.shape[0] if is_array_api_compliant else len(labels)
    if labels_len == 0:
        return 1.0

    pi = xp.astype(xp.unique_counts(labels)[1], _max_precision_float_dtype(xp, device_))

    # single cluster => zero entropy
    if pi.size == 1:
        return 0.0

    pi_sum = xp.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    # Always convert the result as a Python scalar (on CPU) instead of a device
    # specific scalar array.
    return float(-xp.sum((pi / pi_sum) * (xp.log(pi) - log(pi_sum))))
