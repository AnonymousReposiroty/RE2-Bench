def supported_float_dtypes(xp, device=None):
    """Supported floating point types for the namespace.

    Parameters
    ----------
    xp : module
        Array namespace to inspect.

    device : str or device instance from xp, default=None
        Device to use for dtype selection. If ``None``, then a default device
        is assumed.

    Returns
    -------
    supported_dtypes : tuple
        Tuple of real floating data types supported by the provided array namespace,
        ordered from the highest precision to lowest.

    See Also
    --------
    max_precision_float_dtype : Maximum float dtype for a namespace/device pair.

    Notes
    -----
    `float16` is not officially part of the Array API spec at the
    time of writing but scikit-learn estimators and functions can choose
    to accept it when xp.float16 is defined.

    Additionally, some devices available within a namespace may not support
    all floating-point types that the namespace provides.

    https://data-apis.org/array-api/latest/API_specification/data_types.html
    """
    dtypes_dict = xp.__array_namespace_info__().dtypes(
        kind="real floating", device=device
    )
    valid_float_dtypes = []
    for dtype_key in ("float64", "float32"):
        if dtype_key in dtypes_dict:
            valid_float_dtypes.append(dtypes_dict[dtype_key])

    if hasattr(xp, "float16"):
        valid_float_dtypes.append(xp.float16)

    return tuple(valid_float_dtypes)
