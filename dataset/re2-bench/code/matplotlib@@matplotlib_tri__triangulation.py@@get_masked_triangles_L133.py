import sys
import numpy as np
from matplotlib import _qhull

class Triangulation:
    """
    An unstructured triangular grid consisting of npoints points and
    ntri triangles.  The triangles can either be specified by the user
    or automatically generated using a Delaunay triangulation.

    Parameters
    ----------
    x, y : (npoints,) array-like
        Coordinates of grid points.
    triangles : (ntri, 3) array-like of int, optional
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner.  If not
        specified, the Delaunay triangulation is calculated.
    mask : (ntri,) array-like of bool, optional
        Which triangles are masked out.

    Attributes
    ----------
    triangles : (ntri, 3) array of int
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner. If you want to
        take the *mask* into account, use `get_masked_triangles` instead.
    mask : (ntri, 3) array of bool or None
        Masked out triangles.
    is_delaunay : bool
        Whether the Triangulation is a calculated Delaunay
        triangulation (where *triangles* was not specified) or not.

    Notes
    -----
    For a Triangulation to be valid it must not have duplicate points,
    triangles formed from colinear points, or overlapping triangles.
    """

    def __init__(self, x, y, triangles=None, mask=None):
        from matplotlib import _qhull
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise ValueError(f'x and y must be equal-length 1D arrays, but found shapes {self.x.shape!r} and {self.y.shape!r}')
        self.mask = None
        self._edges = None
        self._neighbors = None
        self.is_delaunay = False
        if triangles is None:
            self.triangles, self._neighbors = _qhull.delaunay(x, y, sys.flags.verbose)
            self.is_delaunay = True
        else:
            try:
                self.triangles = np.array(triangles, dtype=np.int32, order='C')
            except ValueError as e:
                raise ValueError(f'triangles must be a (N, 3) int array, not {triangles!r}') from e
            if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
                raise ValueError(f'triangles must be a (N, 3) int array, but found shape {self.triangles.shape!r}')
            if self.triangles.max() >= len(self.x):
                raise ValueError(f'triangles are indices into the points and must be in the range 0 <= i < {len(self.x)} but found value {self.triangles.max()}')
            if self.triangles.min() < 0:
                raise ValueError(f'triangles are indices into the points and must be in the range 0 <= i < {len(self.x)} but found value {self.triangles.min()}')
        self._cpp_triangulation = None
        self._trifinder = None
        self.set_mask(mask)

    def get_masked_triangles(self):
        """
        Return an array of triangles taking the mask into account.
        """
        if self.mask is not None:
            return self.triangles[~self.mask]
        else:
            return self.triangles

    def set_mask(self, mask):
        """
        Set or clear the mask array.

        Parameters
        ----------
        mask : None or bool array of length ntri
        """
        if mask is None:
            self.mask = None
        else:
            self.mask = np.asarray(mask, dtype=bool)
            if self.mask.shape != (self.triangles.shape[0],):
                raise ValueError('mask array must have same length as triangles array')
        if self._cpp_triangulation is not None:
            self._cpp_triangulation.set_mask(self.mask if self.mask is not None else ())
        self._edges = None
        self._neighbors = None
        if self._trifinder is not None:
            self._trifinder._initialize()
