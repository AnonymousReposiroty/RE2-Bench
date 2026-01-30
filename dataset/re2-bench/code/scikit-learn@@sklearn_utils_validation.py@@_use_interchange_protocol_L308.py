def _use_interchange_protocol(X):
    """Use interchange protocol for non-pandas dataframes that follow the protocol.

    Note: at this point we chose not to use the interchange API on pandas dataframe
    to ensure strict behavioral backward compatibility with older versions of
    scikit-learn.
    """
    return not _is_pandas_df(X) and hasattr(X, "__dataframe__")
