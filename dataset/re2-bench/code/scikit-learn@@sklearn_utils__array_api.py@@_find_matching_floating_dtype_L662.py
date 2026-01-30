def _find_matching_floating_dtype(*arrays, xp):
    """Find a suitable floating point dtype when computing with arrays.

    If any of the arrays are floating point, return the dtype with the highest
    precision by following official type promotion rules:

    https://data-apis.org/array-api/latest/API_specification/type_promotion.html

    If there are no floating point input arrays (all integral inputs for
    instance), return the default floating point dtype for the namespace.
    """
    dtyped_arrays = [xp.asarray(a) for a in arrays if hasattr(a, "dtype")]
    floating_dtypes = [
        a.dtype for a in dtyped_arrays if xp.isdtype(a.dtype, "real floating")
    ]
    if floating_dtypes:
        # Return the floating dtype with the highest precision:
        return xp.result_type(*floating_dtypes)

    # If none of the input arrays have a floating point dtype, they must be all
    # integer arrays or containers of Python scalars: return the default
    # floating point dtype for the namespace (implementation specific).
    return xp.asarray(0.0).dtype
