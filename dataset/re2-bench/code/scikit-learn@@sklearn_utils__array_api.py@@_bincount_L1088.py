import numpy

def _bincount(array, weights=None, minlength=None, xp=None):
    # TODO: update if bincount is ever adopted in a future version of the standard:
    # https://github.com/data-apis/array-api/issues/812
    xp, _ = get_namespace(array, xp=xp)
    if hasattr(xp, "bincount"):
        return xp.bincount(array, weights=weights, minlength=minlength)

    array_np = _convert_to_numpy(array, xp=xp)
    if weights is not None:
        weights_np = _convert_to_numpy(weights, xp=xp)
    else:
        weights_np = None
    bin_out = numpy.bincount(array_np, weights=weights_np, minlength=minlength)
    return xp.asarray(bin_out, device=device(array))
