def _validate_diagonal_args(array, value, xp):
    """Validate arguments to `_fill_diagonal`/`_add_to_diagonal`."""
    if array.ndim != 2:
        raise ValueError(
            f"`array` should be 2D. Got array with shape {tuple(array.shape)}"
        )

    value = xp.asarray(value, dtype=array.dtype, device=device(array))
    if value.ndim not in [0, 1]:
        raise ValueError(
            "`value` needs to be a scalar or a 1D array, "
            f"got a {value.ndim}D array instead."
        )
    min_rows_columns = min(array.shape)
    if value.ndim == 1 and value.shape[0] != min_rows_columns:
        raise ValueError(
            "`value` needs to be a scalar or 1D array of the same length as the "
            f"diagonal of `array` ({min_rows_columns}). Got {value.shape[0]}"
        )

    return value, min_rows_columns
