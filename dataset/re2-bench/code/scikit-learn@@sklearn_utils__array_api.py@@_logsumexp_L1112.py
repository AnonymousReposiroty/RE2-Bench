def _logsumexp(array, axis=None, xp=None):
    # TODO replace by scipy.special.logsumexp when
    # https://github.com/scipy/scipy/pull/22683 is part of a release.
    # The following code is strongly inspired and simplified from
    # scipy.special._logsumexp.logsumexp
    xp, _, device = get_namespace_and_device(array, xp=xp)
    axis = tuple(range(array.ndim)) if axis is None else axis

    supported_dtypes = supported_float_dtypes(xp)
    if array.dtype not in supported_dtypes:
        array = xp.asarray(array, dtype=supported_dtypes[0])

    array_max = xp.max(array, axis=axis, keepdims=True)
    index_max = array == array_max

    array = xp.asarray(array, copy=True)
    array[index_max] = -xp.inf
    i_max_dt = xp.astype(index_max, array.dtype)
    m = xp.sum(i_max_dt, axis=axis, keepdims=True, dtype=array.dtype)
    # Specifying device explicitly is the fix for https://github.com/scipy/scipy/issues/22680
    shift = xp.where(
        xp.isfinite(array_max),
        array_max,
        xp.asarray(0, dtype=array_max.dtype, device=device),
    )
    exp = xp.exp(array - shift)
    s = xp.sum(exp, axis=axis, keepdims=True, dtype=exp.dtype)
    s = xp.where(s == 0, s, s / m)
    out = xp.log1p(s) + xp.log(m) + array_max
    out = xp.squeeze(out, axis=axis)
    out = out[()] if out.ndim == 0 else out

    return out
