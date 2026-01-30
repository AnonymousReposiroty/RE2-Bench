import numpy

def _nanmean(X, axis=None, xp=None):
    # TODO: refactor once nan-aware reductions are standardized:
    # https://github.com/data-apis/array-api/issues/621
    xp, _, device_ = get_namespace_and_device(X, xp=xp)
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.nanmean(X, axis=axis))
    else:
        mask = xp.isnan(X)
        total = xp.sum(
            xp.where(mask, xp.asarray(0.0, dtype=X.dtype, device=device_), X), axis=axis
        )
        count = xp.sum(xp.astype(xp.logical_not(mask), X.dtype), axis=axis)
        return total / count
