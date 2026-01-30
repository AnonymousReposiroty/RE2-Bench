def request_is_valid(item):
    """Check if an item is a valid request value (and not an alias).

    Parameters
    ----------
    item : object
        The given item to be checked.

    Returns
    -------
    result : bool
        Whether the given item is valid.
    """
    return item in VALID_REQUEST_VALUES
