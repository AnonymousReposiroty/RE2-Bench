def _add_to_diagonal(array, value, xp):
    """Add `value` to diagonal of `array`.

    Related to `fill_diagonal`. `value` should be a scalar or
    1D of greater or equal length as the diagonal (i.e., `value` is never repeated
    when shorter).

    Note `array` is altered in place.
    """
    value, min_rows_columns = _validate_diagonal_args(array, value, xp)

    if _is_numpy_namespace(xp):
        step = array.shape[1] + 1
        # Ensure we do not wrap
        end = array.shape[1] * array.shape[1]
        array.flat[:end:step] += value
        return

    # TODO: when array libraries support `reshape(copy)`, use
    # `reshape(array, (-1,), copy=False)`, then fill with `[:end:step]` (within
    # `try/except`). This is faster than for loop, when no copy needs to be
    # made within `reshape`. See #31445 for details.
    value = xp.linalg.diagonal(array) + value
    for i in range(min_rows_columns):
        array[i, i] = value[i]
