import sys

def _is_tensorflow_array(x):
    """Return whether *x* is a TensorFlow Tensor or Variable."""
    try:
        # We're intentionally not attempting to import TensorFlow. If somebody
        # has created a TensorFlow array, TensorFlow should already be in
        # sys.modules we use `is_tensor` to not depend on the class structure
        # of TensorFlow arrays, as `tf.Variables` are not instances of
        # `tf.Tensor` (they both convert the same way).
        is_tensor = sys.modules.get("tensorflow").is_tensor
    except AttributeError:
        return False
    try:
        return is_tensor(x)
    except Exception:
        return False  # Just in case it's a very nonstandard module.
