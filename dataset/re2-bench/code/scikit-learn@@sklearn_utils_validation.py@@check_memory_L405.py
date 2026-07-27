import joblib

def check_memory(memory):
    """Check that ``memory`` is joblib.Memory-like.

    joblib.Memory-like means that ``memory`` can be converted into a
    joblib.Memory instance (typically a str denoting the ``location``)
    or has the same interface (has a ``cache`` method).

    Parameters
    ----------
    memory : None, str or object with the joblib.Memory interface
        - If string, the location where to create the `joblib.Memory` interface.
        - If None, no caching is done and the Memory object is completely transparent.

    Returns
    -------
    memory : object with the joblib.Memory interface
        A correct joblib.Memory object.

    Raises
    ------
    ValueError
        If ``memory`` is not joblib.Memory-like.

    Examples
    --------
    >>> from sklearn.utils.validation import check_memory
    >>> check_memory("caching_dir")
    Memory(location=caching_dir/joblib)
    """
    if memory is None or isinstance(memory, str):
        memory = joblib.Memory(location=memory, verbose=0)
    elif not hasattr(memory, "cache"):
        raise ValueError(
            "'memory' should be None, a string or have the same"
            " interface as joblib.Memory."
            " Got memory='{}' instead.".format(memory)
        )
    return memory
