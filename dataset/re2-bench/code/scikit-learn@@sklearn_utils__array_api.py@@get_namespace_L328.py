from sklearn._config import get_config
from sklearn.externals import array_api_compat
from sklearn.externals.array_api_compat import numpy as np_compat

def get_namespace(*arrays, remove_none=True, remove_types=(str,), xp=None):
    """Get namespace of arrays.

    Introspect `arrays` arguments and return their common Array API compatible
    namespace object, if any.

    Note that sparse arrays are filtered by default.

    See: https://numpy.org/neps/nep-0047-array-api-standard.html

    If `arrays` are regular numpy arrays, `array_api_compat.numpy` is returned instead.

    Namespace support is not enabled by default. To enabled it call:

      sklearn.set_config(array_api_dispatch=True)

    or:

      with sklearn.config_context(array_api_dispatch=True):
          # your code here

    Otherwise `array_api_compat.numpy` is
    always returned irrespective of the fact that arrays implement the
    `__array_namespace__` protocol or not.

    Note that if no arrays pass the set filters, ``_NUMPY_API_WRAPPER_INSTANCE, False``
    is returned.

    Parameters
    ----------
    *arrays : array objects
        Array objects.

    remove_none : bool, default=True
        Whether to ignore None objects passed in arrays.

    remove_types : tuple or list, default=(str,)
        Types to ignore in the arrays.

    xp : module, default=None
        Precomputed array namespace module. When passed, typically from a caller
        that has already performed inspection of its own inputs, skips array
        namespace inspection.

    Returns
    -------
    namespace : module
        Namespace shared by array objects. If any of the `arrays` are not arrays,
        the namespace defaults to the NumPy namespace.

    is_array_api_compliant : bool
        True if the arrays are containers that implement the array API spec (see
        https://data-apis.org/array-api/latest/index.html).
        Always False when array_api_dispatch=False.
    """
    array_api_dispatch = get_config()["array_api_dispatch"]
    if not array_api_dispatch:
        if xp is not None:
            return xp, False
        else:
            return np_compat, False

    if xp is not None:
        return xp, True

    arrays = _remove_non_arrays(
        *arrays,
        remove_none=remove_none,
        remove_types=remove_types,
    )

    if not arrays:
        return np_compat, False

    _check_array_api_dispatch(array_api_dispatch)

    namespace, is_array_api_compliant = array_api_compat.get_namespace(*arrays), True

    if namespace.__name__ == "array_api_strict" and hasattr(
        namespace, "set_array_api_strict_flags"
    ):
        namespace.set_array_api_strict_flags(api_version="2024.12")

    return namespace, is_array_api_compliant
