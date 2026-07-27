import pandas

def check_pandas_support(caller_name):
    """Raise ImportError with detailed error message if pandas is not installed.

    Plot utilities like :func:`fetch_openml` should lazily import
    pandas and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires pandas.

    Returns
    -------
    pandas
        The pandas package.
    """
    try:
        import pandas

        return pandas
    except ImportError as e:
        raise ImportError("{} requires pandas.".format(caller_name)) from e
