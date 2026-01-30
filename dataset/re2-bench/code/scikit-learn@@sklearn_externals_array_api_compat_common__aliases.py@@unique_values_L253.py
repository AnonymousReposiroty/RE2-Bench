from ._typing import Array, Device, DType, Namespace

def unique_values(x: Array, /, xp: Namespace) -> Array:
    kwargs = _unique_kwargs(xp)
    return xp.unique(
        x,
        return_counts=False,
        return_index=False,
        return_inverse=False,
        **kwargs,
    )
