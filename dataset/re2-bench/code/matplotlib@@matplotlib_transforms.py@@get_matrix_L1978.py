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

    def get_matrix(self):
        """
        Get the underlying transformation matrix as a 3x3 array::

          a c e
          b d f
          0 0 1

        .
        """
        if self._invalid:
            self._inverted = None
            self._invalid = 0
        return self._mtx
