from sklearn.utils._array_api import (
    _average,
    _find_matching_floating_dtype,
    _median,
    get_namespace,
    get_namespace_and_device,
    size,
)
from sklearn.utils._array_api import _xlogy as xlogy

def _mean_tweedie_deviance(y_true, y_pred, sample_weight, power):
    """Mean Tweedie deviance regression loss."""
    xp, _ = get_namespace(y_true, y_pred)
    p = power
    if p < 0:
        # 'Extreme stable', y any real number, y_pred > 0
        dev = 2 * (
            xp.pow(
                xp.where(y_true > 0, y_true, 0.0),
                2 - p,
            )
            / ((1 - p) * (2 - p))
            - y_true * xp.pow(y_pred, 1 - p) / (1 - p)
            + xp.pow(y_pred, 2 - p) / (2 - p)
        )
    elif p == 0:
        # Normal distribution, y and y_pred any real number
        dev = (y_true - y_pred) ** 2
    elif p == 1:
        # Poisson distribution
        dev = 2 * (xlogy(y_true, y_true / y_pred) - y_true + y_pred)
    elif p == 2:
        # Gamma distribution
        dev = 2 * (xp.log(y_pred / y_true) + y_true / y_pred - 1)
    else:
        dev = 2 * (
            xp.pow(y_true, 2 - p) / ((1 - p) * (2 - p))
            - y_true * xp.pow(y_pred, 1 - p) / (1 - p)
            + xp.pow(y_pred, 2 - p) / (2 - p)
        )
    return float(_average(dev, weights=sample_weight, xp=xp))
