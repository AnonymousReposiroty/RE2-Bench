import inspect
from sklearn.utils._array_api import (
    _average,
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _nanmean,
    _nansum,
    device,
    get_namespace,
    get_namespace_and_device,
)

def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides array accumulator functions with a maximum floating
    precision dtype, usually float64, when used on a floating point input. This
    prevents accumulator overflow on smaller floating point dtypes.

    Parameters
    ----------
    op : function
        An array accumulator function such as np.mean or np.sum.
    x : array
        An array to which the accumulator function is applied.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.

    Returns
    -------
    result
        The output of the accumulator function passed to this function.

    Notes
    -----
    When using array-api support, the accumulator function will upcast floating-point
    arguments to the maximum precision possible for the array namespace and device.
    This is usually float64, but may be float32 for some namespace/device pairs.
    """
    xp, _, x_device = get_namespace_and_device(x)
    max_float_dtype = _max_precision_float_dtype(xp, device=x_device)
    if (
        xp.isdtype(x.dtype, "real floating")
        and xp.finfo(x.dtype).bits < xp.finfo(max_float_dtype).bits
    ):
        # We need to upcast. Some ops support this natively; others don't.
        target_dtype = _max_precision_float_dtype(xp, device=x_device)

        def convert_dtype(arr):
            return xp.astype(arr, target_dtype, copy=False)

        if "dtype" in inspect.signature(op).parameters:
            return op(x, *args, **kwargs, dtype=target_dtype)
        else:
            # This op doesn't support a dtype kwarg, it seems. Rely on manual
            # type promotion, at the cost of memory allocations.
            # xp.matmul is the most commonly used op that lacks a dtype kwarg at
            # the time of writing.
            x = convert_dtype(x)
            args = [
                (convert_dtype(arg) if hasattr(arg, "dtype") else arg) for arg in args
            ]
    return op(x, *args, **kwargs)
