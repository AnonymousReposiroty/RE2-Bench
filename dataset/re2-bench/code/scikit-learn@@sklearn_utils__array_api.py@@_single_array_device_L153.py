from sklearn._config import get_config

def _single_array_device(array):
    """Hardware device where the array data resides on."""
    if (
        not hasattr(array, "device")
        # When array API dispatch is disabled, we expect the scikit-learn code
        # to use np.asarray so that the resulting NumPy array will implicitly use the
        # CPU. In this case, scikit-learn should stay as device neutral as possible,
        # hence the use of `device=None` which is accepted by all libraries, before
        # and after the expected conversion to NumPy via np.asarray.
        or not get_config()["array_api_dispatch"]
    ):
        return None
    else:
        return array.device
