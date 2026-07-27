import numpy as np

class Affine2D(Affine2DBase):
    """
    A mutable 2D affine transformation.
    """

    def __init__(self, matrix=None, **kwargs):
        """
        Initialize an Affine transform from a 3x3 numpy float array::

          a c e
          b d f
          0 0 1

        If *matrix* is None, initialize with the identity transform.
        """
        super().__init__(**kwargs)
        if matrix is None:
            matrix = IdentityTransform._mtx
        self._mtx = matrix.copy()
        self._invalid = 0
    _base_str = _make_str_method('_mtx')

    def __str__(self):
        return self._base_str() if (self._mtx != np.diag(np.diag(self._mtx))).any() else f'Affine2D().scale({self._mtx[0, 0]}, {self._mtx[1, 1]})' if self._mtx[0, 0] != self._mtx[1, 1] else f'Affine2D().scale({self._mtx[0, 0]})'
