import numpy

def _nansum(X, axis=None, xp=None, keepdims=False, dtype=None):
    # TODO: refactor once nan-aware reductions are standardized:
    # https://github.com/data-apis/array-api/issues/621
    xp, _, X_device = get_namespace_and_device(X, xp=xp)

    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.nansum(X, axis=axis, keepdims=keepdims, dtype=dtype))

    mask = xp.isnan(X)
    masked_arr = xp.where(mask, xp.asarray(0, device=X_device, dtype=X.dtype), X)
    return xp.sum(masked_arr, axis=axis, keepdims=keepdims, dtype=dtype)
