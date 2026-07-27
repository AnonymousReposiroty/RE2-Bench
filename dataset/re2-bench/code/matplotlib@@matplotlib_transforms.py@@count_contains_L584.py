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
    def min(self):
        """The bottom-left corner of the bounding box."""
        return np.min(self.get_points(), axis=0)

    @property
    def max(self):
        """The top-right corner of the bounding box."""
        return np.max(self.get_points(), axis=0)

    def get_points(self):
        raise NotImplementedError
    coefs = {'C': (0.5, 0.5), 'SW': (0, 0), 'S': (0.5, 0), 'SE': (1.0, 0), 'E': (1.0, 0.5), 'NE': (1.0, 1.0), 'N': (0.5, 1.0), 'NW': (0, 1.0), 'W': (0, 0.5)}

    def count_contains(self, vertices):
        """
        Count the number of vertices contained in the `Bbox`.
        Any vertices with a non-finite x or y value are ignored.

        Parameters
        ----------
        vertices : (N, 2) array
        """
        if len(vertices) == 0:
            return 0
        vertices = np.asarray(vertices)
        with np.errstate(invalid='ignore'):
            return ((self.min < vertices) & (vertices < self.max)).all(axis=1).sum()
