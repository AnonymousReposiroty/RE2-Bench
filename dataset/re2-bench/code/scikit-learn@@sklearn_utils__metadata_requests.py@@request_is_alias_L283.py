def request_is_alias(item):
    """Check if an item is a valid string alias for a metadata.

    Values in ``VALID_REQUEST_VALUES`` are not considered aliases in this
    context. Only a string which is a valid identifier is.

    Parameters
    ----------
    item : object
        The given item to be checked if it can be an alias for the metadata.

    Returns
    -------
    result : bool
        Whether the given item is a valid alias.
    """
    if item in VALID_REQUEST_VALUES:
        return False

    # item is only an alias if it's a valid identifier
    return isinstance(item, str) and item.isidentifier()
