from ._typing import Array, Device, DType, Namespace

def unique_counts(x: Array, /, xp: Namespace) -> UniqueCountsResult:
    kwargs = _unique_kwargs(xp)
    res = xp.unique(
        x, return_counts=True, return_index=False, return_inverse=False, **kwargs
    )

    return UniqueCountsResult(*res)
