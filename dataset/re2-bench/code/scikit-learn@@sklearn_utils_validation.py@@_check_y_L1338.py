import numpy as np

def _check_y(y, multi_output=False, y_numeric=False, estimator=None):
    """Isolated part of check_X_y dedicated to y validation"""
    if multi_output:
        y = check_array(
            y,
            accept_sparse="csr",
            ensure_all_finite=True,
            ensure_2d=False,
            dtype=None,
            input_name="y",
            estimator=estimator,
        )
    else:
        estimator_name = _check_estimator_name(estimator)
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y, input_name="y", estimator_name=estimator_name)
        _ensure_no_complex_data(y)
    if y_numeric and hasattr(y.dtype, "kind") and y.dtype.kind == "O":
        y = y.astype(np.float64)

    return y
