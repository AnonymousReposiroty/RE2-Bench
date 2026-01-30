import warnings
import numpy as np
from sklearn._config import get_config

def get_chunk_n_rows(row_bytes, *, max_n_rows=None, working_memory=None):
    """Calculate how many rows can be processed within `working_memory`.

    Parameters
    ----------
    row_bytes : int
        The expected number of bytes of memory that will be consumed
        during the processing of each row.
    max_n_rows : int, default=None
        The maximum return value.
    working_memory : int or float, default=None
        The number of rows to fit inside this number of MiB will be
        returned. When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    Returns
    -------
    int
        The number of rows which can be processed within `working_memory`.

    Warns
    -----
    Issues a UserWarning if `row_bytes exceeds `working_memory` MiB.
    """

    if working_memory is None:
        working_memory = get_config()["working_memory"]

    chunk_n_rows = int(working_memory * (2**20) // row_bytes)
    if max_n_rows is not None:
        chunk_n_rows = min(chunk_n_rows, max_n_rows)
    if chunk_n_rows < 1:
        warnings.warn(
            "Could not adhere to working_memory config. "
            "Currently %.0fMiB, %.0fMiB required."
            % (working_memory, np.ceil(row_bytes * 2**-20))
        )
        chunk_n_rows = 1
    return chunk_n_rows
