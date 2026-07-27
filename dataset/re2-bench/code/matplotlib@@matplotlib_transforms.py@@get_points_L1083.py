import numpy as np

class Bbox(BboxBase):
    """
    A mutable bounding box.

    Examples
    --------
    **Create from known bounds**

    The default constructor takes the boundary "points" ``[[xmin, ymin],
    [xmax, ymax]]``.

        >>> Bbox([[1, 1], [3, 7]])
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    Alternatively, a Bbox can be created from the flattened points array, the
    so-called "extents" ``(xmin, ymin, xmax, ymax)``

        >>> Bbox.from_extents(1, 1, 3, 7)
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    or from the "bounds" ``(xmin, ymin, width, height)``.

        >>> Bbox.from_bounds(1, 1, 2, 6)
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    **Create from collections of points**

    The "empty" object for accumulating Bboxs is the null bbox, which is a
    stand-in for the empty set.

        >>> Bbox.null()
        Bbox([[inf, inf], [-inf, -inf]])

    Adding points to the null bbox will give you the bbox of those points.

        >>> box = Bbox.null()
        >>> box.update_from_data_xy([[1, 1]])
        >>> box
        Bbox([[1.0, 1.0], [1.0, 1.0]])
        >>> box.update_from_data_xy([[2, 3], [3, 2]], ignore=False)
        >>> box
        Bbox([[1.0, 1.0], [3.0, 3.0]])

    Setting ``ignore=True`` is equivalent to starting over from a null bbox.

        >>> box.update_from_data_xy([[1, 1]], ignore=True)
        >>> box
        Bbox([[1.0, 1.0], [1.0, 1.0]])

    .. warning::

        It is recommended to always specify ``ignore`` explicitly.  If not, the
        default value of ``ignore`` can be changed at any time by code with
        access to your Bbox, for example using the method `~.Bbox.ignore`.

    **Properties of the ``null`` bbox**

    .. note::

        The current behavior of `Bbox.null()` may be surprising as it does
        not have all of the properties of the "empty set", and as such does
        not behave like a "zero" object in the mathematical sense. We may
        change that in the future (with a deprecation period).

    The null bbox is the identity for intersections

        >>> Bbox.intersection(Bbox([[1, 1], [3, 7]]), Bbox.null())
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    except with itself, where it returns the full space.

        >>> Bbox.intersection(Bbox.null(), Bbox.null())
        Bbox([[-inf, -inf], [inf, inf]])

    A union containing null will always return the full space (not the other
    set!)

        >>> Bbox.union([Bbox([[0, 0], [0, 0]]), Bbox.null()])
        Bbox([[-inf, -inf], [inf, inf]])
    """

    def __init__(self, points, **kwargs):
        """
        Parameters
        ----------
        points : `~numpy.ndarray`
            A (2, 2) array of the form ``[[x0, y0], [x1, y1]]``.
        """
        super().__init__(**kwargs)
        points = np.asarray(points, float)
        if points.shape != (2, 2):
            raise ValueError('Bbox points must be of the form "[[x0, y0], [x1, y1]]".')
        self._points = points
        self._minpos = _default_minpos.copy()
        self._ignore = True
        self._points_orig = self._points.copy()
    if DEBUG:
        ___init__ = __init__

        def __init__(self, points, **kwargs):
            self._check(points)
            self.___init__(points, **kwargs)

        def invalidate(self):
            self._check(self._points)
            super().invalidate()

    def get_points(self):
        """
        Get the points of the bounding box as an array of the form
        ``[[x0, y0], [x1, y1]]``.
        """
        self._invalid = 0
        return self._points
