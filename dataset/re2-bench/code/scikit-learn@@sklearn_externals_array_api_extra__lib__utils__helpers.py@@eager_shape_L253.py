import math
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
)
from ._typing import Array, Device

def eager_shape(x: Array, /) -> tuple[int, ...]:
    """
    Return shape of an array. Raise if shape is not fully defined.

    Parameters
    ----------
    x : Array
        Input array.

    Returns
    -------
    tuple[int, ...]
        Shape of the array.
    """
    shape = x.shape
    # Dask arrays uses non-standard NaN instead of None
    if any(s is None or math.isnan(s) for s in shape):
        msg = "Unsupported lazy shape"
        raise TypeError(msg)
    return cast(tuple[int, ...], shape)
