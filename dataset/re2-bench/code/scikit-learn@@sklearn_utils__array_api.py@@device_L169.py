def device(*array_list, remove_none=True, remove_types=(str,)):
    """Hardware device where the array data resides on.

    If the hardware device is not the same for all arrays, an error is raised.

    Parameters
    ----------
    *array_list : arrays
        List of array instances from NumPy or an array API compatible library.

    remove_none : bool, default=True
        Whether to ignore None objects passed in array_list.

    remove_types : tuple or list, default=(str,)
        Types to ignore in array_list.

    Returns
    -------
    out : device
        `device` object (see the "Device Support" section of the array API spec).
    """
    array_list = _remove_non_arrays(
        *array_list, remove_none=remove_none, remove_types=remove_types
    )

    if not array_list:
        return None

    device_ = _single_array_device(array_list[0])

    # Note: here we cannot simply use a Python `set` as it requires
    # hashable members which is not guaranteed for Array API device
    # objects. In particular, CuPy devices are not hashable at the
    # time of writing.
    for array in array_list[1:]:
        device_other = _single_array_device(array)
        if device_ != device_other:
            raise ValueError(
                f"Input arrays use different devices: {device_}, {device_other}"
            )

    return device_
