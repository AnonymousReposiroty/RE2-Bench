import sys
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
import jax
import jax
import jax.experimental.array_api  # noqa: F401  # pyright: ignore
import jax.numpy

def _is_jax_zero_gradient_array(x: object) -> TypeGuard[_ZeroGradientArray]:
    """Return True if `x` is a zero-gradient array.

    These arrays are a design quirk of Jax that may one day be removed.
    See https://github.com/google/jax/issues/20620.
    """
    # Fast exit
    try:
        dtype = x.dtype  # type: ignore[attr-defined]
    except AttributeError:
        return False
    cls = cast(Hashable, type(dtype))
    if not _issubclass_fast(cls, "numpy.dtypes", "VoidDType"):
        return False

    if "jax" not in sys.modules:
        return False

    import jax
    # jax.float0 is a np.dtype([('float0', 'V')])
    return dtype == jax.float0
