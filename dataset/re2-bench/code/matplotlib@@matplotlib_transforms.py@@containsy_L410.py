import numpy as np
from matplotlib import _api

class BboxBase(TransformNode):
    """
    The base class of all bounding boxes.

    This class is immutable; `Bbox` is a mutable subclass.

    The canonical representation is as two points, with no
    restrictions on their ordering.  Convenience properties are
    provided to get the left, bottom, right and top edges and width
    and height, but these are not stored explicitly.
    """
    is_affine = True
    if DEBUG:

        @staticmethod
        def _check(points):
            if isinstance(points, np.ma.MaskedArray):
                _api.warn_external('Bbox bounds are a masked array.')
            points = np.asarray(points)
            if any(points[1, :] - points[0, :] == 0):
                _api.warn_external('Singular Bbox.')
    frozen.__doc__ = TransformNode.__doc__

    @property
    def intervaly(self):
        """
        The pair of *y* coordinates that define the bounding box.

        This is not guaranteed to be sorted from bottom to top.
        """
        return self.get_points()[:, 1]

    def get_points(self):
        raise NotImplementedError

    def containsy(self, y):
        """
        Return whether *y* is in the closed (:attr:`y0`, :attr:`y1`) interval.
        """
        y0, y1 = self.intervaly
        return y0 <= y <= y1 or y0 >= y >= y1
    coefs = {'C': (0.5, 0.5), 'SW': (0, 0), 'S': (0.5, 0), 'SE': (1.0, 0), 'E': (1.0, 0.5), 'NE': (1.0, 1.0), 'N': (0.5, 1.0), 'NW': (0, 1.0), 'W': (0, 0.5)}
