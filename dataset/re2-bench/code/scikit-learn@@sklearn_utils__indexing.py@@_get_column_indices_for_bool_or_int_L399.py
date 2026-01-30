import numpy as np

def _get_column_indices_for_bool_or_int(key, n_columns):
    # Convert key into list of positive integer indexes
    try:
        idx = _safe_indexing(np.arange(n_columns), key)
    except IndexError as e:
        raise ValueError(
            f"all features must be in [0, {n_columns - 1}] or [-{n_columns}, 0]"
        ) from e
    return np.atleast_1d(idx).tolist()
