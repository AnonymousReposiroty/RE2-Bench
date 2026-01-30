import sys

def _is_pandas_df(X):
    """Return True if the X is a pandas dataframe."""
    try:
        pd = sys.modules["pandas"]
    except KeyError:
        return False
    return isinstance(X, pd.DataFrame)
