import numpy as np
from matplotlib import _api

class Affine2DBase(AffineBase):
    """
    The base class of all 2D affine transformations.

    2D affine transformations are performed using a 3x3 numpy array::

        a c e
        b d f
        0 0 1

    This class provides the read-only interface.  For a mutable 2D
    affine transformation, use `Affine2D`.

    Subclasses of this class will generally only need to override a
    constructor and `~.Transform.get_matrix` that generates a custom 3x3 matrix.
    """
    input_dims = 2
    output_dims = 2

    def to_values(self):
        """
        Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
        """
        mtx = self.get_matrix()
        return tuple(mtx[:2].swapaxes(0, 1).flat)
    if DEBUG:
        _transform_affine = transform_affine

        def transform_affine(self, values):
            if not isinstance(values, np.ndarray):
                _api.warn_external(f'A non-numpy array of type {type(values)} was passed in for transformation, which results in poor performance.')
            return self._transform_affine(values)
