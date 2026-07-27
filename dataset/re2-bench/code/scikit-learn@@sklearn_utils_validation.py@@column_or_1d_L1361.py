import warnings
from sklearn.exceptions import (
    DataConversionWarning,
    NotFittedError,
    PositiveSpectrumWarning,
)
from sklearn.utils._array_api import (
    _asarray_with_order,
    _convert_to_numpy,
    _is_numpy_namespace,
    _max_precision_float_dtype,
    get_namespace,
    get_namespace_and_device,
)

def column_or_1d(y, *, dtype=None, input_name="y", warn=False, device=None):
    """Ravel column or 1d numpy array, else raises an error.

    Parameters
    ----------
    y : array-like
       Input data.

    dtype : data-type, default=None
        Data type for `y`.

        .. versionadded:: 1.2

    input_name : str, default="y"
        The data name used to construct the error message.

        .. versionadded:: 1.8

    warn : bool, default=False
       To control display of warnings.

    device : device, default=None
        `device` object.
        See the :ref:`Array API User Guide <array_api>` for more details.

        .. versionadded:: 1.6

    Returns
    -------
    y : ndarray
       Output data.

    Raises
    ------
    ValueError
        If `y` is not a 1D array or a 2D array with a single row or column.

    Examples
    --------
    >>> from sklearn.utils.validation import column_or_1d
    >>> column_or_1d([1, 1])
    array([1, 1])
    """
    xp, _ = get_namespace(y)
    y = check_array(
        y,
        ensure_2d=False,
        dtype=dtype,
        input_name=input_name,
        ensure_all_finite=False,
        ensure_min_samples=0,
    )

    shape = y.shape
    if len(shape) == 1:
        return _asarray_with_order(
            xp.reshape(y, (-1,)), order="C", xp=xp, device=device
        )
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples, ), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )
        return _asarray_with_order(
            xp.reshape(y, (-1,)), order="C", xp=xp, device=device
        )

    raise ValueError(
        "y should be a 1d array, got an array of shape {} instead.".format(shape)
    )
