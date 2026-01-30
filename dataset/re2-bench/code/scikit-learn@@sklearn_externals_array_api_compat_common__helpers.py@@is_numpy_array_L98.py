from collections.abc import Collection, Hashable
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    SupportsIndex,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)
import numpy.typing as npt

def is_numpy_array(x: object) -> TypeGuard[npt.NDArray[Any]]:
    """
    Return True if `x` is a NumPy array.

    This function does not import NumPy if it has not already been imported
    and is therefore cheap to use.

    This also returns True for `ndarray` subclasses and NumPy scalar objects.

    See Also
    --------

    array_namespace
    is_array_api_obj
    is_cupy_array
    is_torch_array
    is_ndonnx_array
    is_dask_array
    is_jax_array
    is_pydata_sparse_array
    """
    # TODO: Should we reject ndarray subclasses?
    cls = cast(Hashable, type(x))
    return (
        _issubclass_fast(cls, "numpy", "ndarray") 
        or _issubclass_fast(cls, "numpy", "generic")
    ) and not _is_jax_zero_gradient_array(x)
