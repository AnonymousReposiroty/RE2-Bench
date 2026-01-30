def _safe_min(items):
    """Compute the minimum of a list of potentially non-comparable values.

    If values cannot be directly compared due to type incompatibility, the object with
    the lowest string representation is returned.
    """
    try:
        return min(items)
    except TypeError as e:
        if "'<' not supported between" in str(e):
            return min(items, key=lambda x: (str(type(x)), str(x)))
        raise  # pragma: no cover
