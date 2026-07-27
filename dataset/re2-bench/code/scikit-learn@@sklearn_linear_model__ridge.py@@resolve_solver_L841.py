import warnings
from sklearn.utils._array_api import (
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _ravel,
    device,
    get_namespace,
    get_namespace_and_device,
    move_to,
)

def resolve_solver(solver, positive, return_intercept, is_sparse, xp):
    if solver != "auto":
        return solver

    is_numpy_namespace = _is_numpy_namespace(xp)

    auto_solver_np = resolve_solver_for_numpy(positive, return_intercept, is_sparse)
    if is_numpy_namespace:
        return auto_solver_np

    if positive:
        raise ValueError(
            "The solvers that support positive fitting do not support "
            f"Array API dispatch to namespace {xp.__name__}. Please "
            "either disable Array API dispatch, or use a numpy-like "
            "namespace, or set `positive=False`."
        )

    # At the moment, Array API dispatch only supports the "svd" solver.
    solver = "svd"
    if solver != auto_solver_np:
        warnings.warn(
            f"Using Array API dispatch to namespace {xp.__name__} with "
            f"`solver='auto'` will result in using the solver '{solver}'. "
            "The results may differ from those when using a Numpy array, "
            f"because in that case the preferred solver would be {auto_solver_np}. "
            f"Set `solver='{solver}'` to suppress this warning."
        )

    return solver
