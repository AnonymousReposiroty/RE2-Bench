from ._helpers import is_cupy_namespace as _is_cupy_namespace
from ._typing import Array, Device, DType, Namespace

def sign(x: Array, /, xp: Namespace, **kwargs: object) -> Array:
    if isdtype(x.dtype, "complex floating", xp=xp):
        out = (x / xp.abs(x, **kwargs))[...]
        # sign(0) = 0 but the above formula would give nan
        out[x == 0j] = 0j
    else:
        out = xp.sign(x, **kwargs)
    # CuPy sign() does not propagate nans. See
    # https://github.com/data-apis/array-api-compat/issues/136
    if _is_cupy_namespace(xp) and isdtype(x.dtype, "real floating", xp=xp):
        out[xp.isnan(x)] = xp.nan
    return out[()]
