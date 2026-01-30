from sklearn.utils._array_api import (
    _find_matching_floating_dtype,
    _max_precision_float_dtype,
    _modify_in_place_if_numpy,
    device,
    get_namespace,
    get_namespace_and_device,
    size,
    supported_float_dtypes,
)

def _is_constant_feature(var, mean, n_samples):
    """Detect if a feature is indistinguishable from a constant feature.

    The detection is based on its computed variance and on the theoretical
    error bounds of the '2 pass algorithm' for variance computation.

    See "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    """
    # In scikit-learn, variance is always computed using float64 accumulators.
    xp, _, device_ = get_namespace_and_device(var, mean)
    max_float_dtype = _max_precision_float_dtype(xp=xp, device=device_)
    eps = xp.finfo(max_float_dtype).eps

    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound
