import warnings
from sklearn.exceptions import ConvergenceWarning

def _check_optimize_result(solver, result, max_iter=None, extra_warning_msg=None):
    """Check the OptimizeResult for successful convergence

    Parameters
    ----------
    solver : str
       Solver name. Currently only `lbfgs` is supported.

    result : OptimizeResult
       Result of the scipy.optimize.minimize function.

    max_iter : int, default=None
       Expected maximum number of iterations.

    extra_warning_msg : str, default=None
        Extra warning message.

    Returns
    -------
    n_iter : int
       Number of iterations.
    """
    # handle both scipy and scikit-learn solver names
    if solver == "lbfgs":
        if max_iter is not None:
            # In scipy <= 1.0.0, nit may exceed maxiter for lbfgs.
            # See https://github.com/scipy/scipy/issues/7854
            n_iter_i = min(result.nit, max_iter)
        else:
            n_iter_i = result.nit

        if result.status != 0:
            warning_msg = (
                f"{solver} failed to converge after {n_iter_i} iteration(s) "
                f"(status={result.status}):\n"
                f"{result.message}\n"
            )
            # Append a recommendation to increase iterations only when the
            # number of iterations reaches the maximum allowed (max_iter),
            # as this suggests the optimization may have been prematurely
            # terminated due to the iteration limit.
            if max_iter is not None and n_iter_i == max_iter:
                warning_msg += (
                    f"\nIncrease the number of iterations to improve the "
                    f"convergence (max_iter={max_iter})."
                )
            warning_msg += (
                "\nYou might also want to scale the data as shown in:\n"
                "    https://scikit-learn.org/stable/modules/"
                "preprocessing.html"
            )
            if extra_warning_msg is not None:
                warning_msg += "\n" + extra_warning_msg
            warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)

    else:
        raise NotImplementedError

    return n_iter_i
