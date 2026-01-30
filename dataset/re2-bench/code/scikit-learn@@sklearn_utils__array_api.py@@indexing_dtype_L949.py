def indexing_dtype(xp):
    """Return a platform-specific integer dtype suitable for indexing.

    On 32-bit platforms, this will typically return int32 and int64 otherwise.

    Note: using dtype is recommended for indexing transient array
    datastructures. For long-lived arrays, such as the fitted attributes of
    estimators, it is instead recommended to use platform-independent int32 if
    we do not expect to index more 2B elements. Using fixed dtypes simplifies
    the handling of serialized models, e.g. to deploy a model fit on a 64-bit
    platform to a target 32-bit platform such as WASM/pyodide.
    """
    # Currently this is implemented with simple hack that assumes that
    # following "may be" statements in the Array API spec always hold:
    # > The default integer data type should be the same across platforms, but
    # > the default may vary depending on whether Python is 32-bit or 64-bit.
    # > The default array index data type may be int32 on 32-bit platforms, but
    # > the default should be int64 otherwise.
    # https://data-apis.org/array-api/latest/API_specification/data_types.html#default-data-types
    # TODO: once sufficiently adopted, we might want to instead rely on the
    # newer inspection API: https://github.com/data-apis/array-api/issues/640
    return xp.asarray(0).dtype
