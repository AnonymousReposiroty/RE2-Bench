import os
import scipy
from sklearn.utils.fixes import parse_version

def _check_array_api_dispatch(array_api_dispatch):
    """Checks that array API support is functional.

    In particular scipy needs to be recent enough and the environment variable
    needs to be set: SCIPY_ARRAY_API=1.
    """
    if not array_api_dispatch:
        return

    scipy_version = parse_version(scipy.__version__)
    min_scipy_version = "1.14.0"
    if scipy_version < parse_version(min_scipy_version):
        raise ImportError(
            f"SciPy must be {min_scipy_version} or newer"
            " (found {scipy.__version__}) to dispatch array using"
            " the array API specification"
        )

    if os.environ.get("SCIPY_ARRAY_API") != "1":
        raise RuntimeError(
            "Scikit-learn array API support was enabled but scipy's own support is "
            "not enabled. Please set the SCIPY_ARRAY_API=1 environment variable "
            "before importing sklearn or scipy. More details at: "
            "https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html"
        )
