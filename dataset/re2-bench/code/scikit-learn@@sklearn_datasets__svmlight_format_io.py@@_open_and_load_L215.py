from contextlib import closing
import numpy as np
from sklearn.datasets._svmlight_format_fast import (
    _dump_svmlight_file,
    _load_svmlight_file,
)

def _open_and_load(f, dtype, multilabel, zero_based, query_id, offset=0, length=-1):
    if hasattr(f, "read"):
        actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(
            f, dtype, multilabel, zero_based, query_id, offset, length
        )
    else:
        with closing(_gen_open(f)) as f:
            actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(
                f, dtype, multilabel, zero_based, query_id, offset, length
            )

    # convert from array.array, give data the right dtype
    if not multilabel:
        labels = np.frombuffer(labels, np.float64)
    data = np.frombuffer(data, actual_dtype)
    indices = np.frombuffer(ind, np.longlong)
    indptr = np.frombuffer(indptr, dtype=np.longlong)  # never empty
    query = np.frombuffer(query, np.int64)

    data = np.asarray(data, dtype=dtype)  # no-op for float{32,64}
    return data, indices, indptr, labels, query
