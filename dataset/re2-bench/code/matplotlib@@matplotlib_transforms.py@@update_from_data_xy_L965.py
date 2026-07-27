import numpy as np
from .path import Path

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

    def update_from_path(self, path, ignore=None, updatex=True, updatey=True):
        """
        Update the bounds of the `Bbox` to contain the vertices of the
        provided path. After updating, the bounds will have positive *width*
        and *height*; *x0* and *y0* will be the minimal values.

        Parameters
        ----------
        path : `~matplotlib.path.Path`
        ignore : bool, optional
           - When ``True``, ignore the existing bounds of the `Bbox`.
           - When ``False``, include the existing bounds of the `Bbox`.
           - When ``None``, use the last value passed to :meth:`ignore`.
        updatex, updatey : bool, default: True
            When ``True``, update the x/y values.
        """
        if ignore is None:
            ignore = self._ignore
        if path.vertices.size == 0 or not (updatex or updatey):
            return
        if ignore:
            points = np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
            minpos = np.array([np.inf, np.inf])
        else:
            points = self._points.copy()
            minpos = self._minpos.copy()
        valid_points = np.isfinite(path.vertices[..., 0]) & np.isfinite(path.vertices[..., 1])
        if updatex:
            x = path.vertices[..., 0][valid_points]
            points[0, 0] = min(points[0, 0], np.min(x, initial=np.inf))
            points[1, 0] = max(points[1, 0], np.max(x, initial=-np.inf))
            minpos[0] = min(minpos[0], np.min(x[x > 0], initial=np.inf))
        if updatey:
            y = path.vertices[..., 1][valid_points]
            points[0, 1] = min(points[0, 1], np.min(y, initial=np.inf))
            points[1, 1] = max(points[1, 1], np.max(y, initial=-np.inf))
            minpos[1] = min(minpos[1], np.min(y[y > 0], initial=np.inf))
        if np.any(points != self._points) or np.any(minpos != self._minpos):
            self.invalidate()
            if updatex:
                self._points[:, 0] = points[:, 0]
                self._minpos[0] = minpos[0]
            if updatey:
                self._points[:, 1] = points[:, 1]
                self._minpos[1] = minpos[1]

    def update_from_data_xy(self, xy, ignore=None, updatex=True, updatey=True):
        """
        Update the `Bbox` bounds based on the passed in *xy* coordinates.

        After updating, the bounds will have positive *width* and *height*;
        *x0* and *y0* will be the minimal values.

        Parameters
        ----------
        xy : (N, 2) array-like
            The (x, y) coordinates.
        ignore : bool, optional
            - When ``True``, ignore the existing bounds of the `Bbox`.
            - When ``False``, include the existing bounds of the `Bbox`.
            - When ``None``, use the last value passed to :meth:`ignore`.
        updatex, updatey : bool, default: True
             When ``True``, update the x/y values.
        """
        if len(xy) == 0:
            return
        path = Path(xy)
        self.update_from_path(path, ignore=ignore, updatex=updatex, updatey=updatey)
