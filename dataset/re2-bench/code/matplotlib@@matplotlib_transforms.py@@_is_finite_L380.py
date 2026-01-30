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
    def width(self):
        """The (signed) width of the bounding box."""
        points = self.get_points()
        return points[1, 0] - points[0, 0]

    @property
    def height(self):
        """The (signed) height of the bounding box."""
        points = self.get_points()
        return points[1, 1] - points[0, 1]

    def get_points(self):
        raise NotImplementedError

    def _is_finite(self):
        """
        Return whether the bounding box is finite and not degenerate to a
        single point.

        We count the box as finite if neither width nor height are infinite
        and at least one direction is non-zero; i.e. a point is not finite,
        but a horizontal or vertical line is.

        .. versionadded:: 3.11

        Notes
        -----
        We keep this private for now because concise naming is hard and
        because we are not sure how universal the concept is. It is
        currently used only for filtering bboxes to be included in
        tightbbox calculation, but I'm unsure whether single points
        should be included there as well.
        """
        width = self.width
        height = self.height
        return (width > 0 or height > 0) and width < np.inf and (height < np.inf)
    coefs = {'C': (0.5, 0.5), 'SW': (0, 0), 'S': (0.5, 0), 'SE': (1.0, 0), 'E': (1.0, 0.5), 'NE': (1.0, 1.0), 'N': (0.5, 1.0), 'NW': (0, 1.0), 'W': (0, 0.5)}
