import numpy

def _nanmin(X, axis=None, xp=None):
    # TODO: refactor once nan-aware reductions are standardized:
    # https://github.com/data-apis/array-api/issues/621
    xp, _, device_ = get_namespace_and_device(X, xp=xp)
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.nanmin(X, axis=axis))

    else:
        mask = xp.isnan(X)
        X = xp.min(
            xp.where(mask, xp.asarray(+xp.inf, dtype=X.dtype, device=device_), X),
            axis=axis,
        )
        # Replace Infs from all NaN slices with NaN again
        mask = xp.all(mask, axis=axis)
        if xp.any(mask):
            X = xp.where(mask, xp.asarray(xp.nan, dtype=X.dtype, device=device_), X)
        return X
