import math
import warnings

def _safe_divide(numerator, denominator):
    """Prevents overflow and division by zero."""
    # This is used for classifiers where the denominator might become zero exactly.
    # For instance for log loss, HalfBinomialLoss, if proba=0 or proba=1 exactly, then
    # denominator = hessian = 0, and we should set the node value in the line search to
    # zero as there is no improvement of the loss possible.
    # For numerical safety, we do this already for extremely tiny values.
    if abs(denominator) < 1e-150:
        return 0.0
    else:
        # Cast to Python float to trigger Python errors, e.g. ZeroDivisionError,
        # without relying on `np.errstate` that is not supported by Pyodide.
        result = float(numerator) / float(denominator)
        # Cast to Python float to trigger a ZeroDivisionError without relying
        # on `np.errstate` that is not supported by Pyodide.
        result = float(numerator) / float(denominator)
        if math.isinf(result):
            warnings.warn("overflow encountered in _safe_divide", RuntimeWarning)
        return result
