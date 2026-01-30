import numpy as np
from matplotlib import cbook

class Registry(dict):
    """Register types with conversion interface."""

    def get_converter(self, x):
        """Get the converter interface instance for *x*, or None."""
        x = cbook._unpack_to_numpy(x)
        if isinstance(x, np.ndarray):
            x = np.ma.getdata(x).ravel()
            if not x.size:
                return self.get_converter(np.array([0], dtype=x.dtype))
        for cls in type(x).__mro__:
            try:
                return self[cls]
            except KeyError:
                pass
        try:
            first = cbook._safe_first_finite(x)
        except (TypeError, StopIteration):
            pass
        else:
            if type(first) is not type(x):
                return self.get_converter(first)
        return None
