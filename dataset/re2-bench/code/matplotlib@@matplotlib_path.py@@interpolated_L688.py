from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation

class Path:
    """
    A series of possibly disconnected, possibly closed, line and curve
    segments.

    The underlying storage is made up of two parallel numpy arrays:

    - *vertices*: an (N, 2) float array of vertices
    - *codes*: an N-length `numpy.uint8` array of path codes, or None

    These two arrays always have the same length in the first
    dimension.  For example, to represent a cubic curve, you must
    provide three vertices and three `CURVE4` codes.

    The code types are:

    - `STOP`   :  1 vertex (ignored)
        A marker for the end of the entire path (currently not required and
        ignored)

    - `MOVETO` :  1 vertex
        Pick up the pen and move to the given vertex.

    - `LINETO` :  1 vertex
        Draw a line from the current position to the given vertex.

    - `CURVE3` :  1 control point, 1 endpoint
        Draw a quadratic Bézier curve from the current position, with the given
        control point, to the given end point.

    - `CURVE4` :  2 control points, 1 endpoint
        Draw a cubic Bézier curve from the current position, with the given
        control points, to the given end point.

    - `CLOSEPOLY` : 1 vertex (ignored)
        Draw a line segment to the start point of the current polyline.

    If *codes* is None, it is interpreted as a `MOVETO` followed by a series
    of `LINETO`.

    Users of Path objects should not access the vertices and codes arrays
    directly.  Instead, they should use `iter_segments` or `cleaned` to get the
    vertex/code pairs.  This helps, in particular, to consistently handle the
    case of *codes* being None.

    Some behavior of Path objects can be controlled by rcParams. See the
    rcParams whose keys start with 'path.'.

    .. note::

        The vertices and codes arrays should be treated as
        immutable -- there are a number of optimizations and assumptions
        made up front in the constructor that will not change when the
        data changes.
    """
    code_type = np.uint8
    STOP = code_type(0)
    MOVETO = code_type(1)
    LINETO = code_type(2)
    CURVE3 = code_type(3)
    CURVE4 = code_type(4)
    CLOSEPOLY = code_type(79)
    NUM_VERTICES_FOR_CODE = {STOP: 1, MOVETO: 1, LINETO: 1, CURVE3: 2, CURVE4: 3, CLOSEPOLY: 1}

    def __init__(self, vertices, codes=None, _interpolation_steps=1, closed=False, readonly=False):
        """
        Create a new path with the given vertices and codes.

        Parameters
        ----------
        vertices : (N, 2) array-like
            The path vertices, as an array, masked array or sequence of pairs.
            Masked values, if any, will be converted to NaNs, which are then
            handled correctly by the Agg PathIterator and other consumers of
            path data, such as :meth:`iter_segments`.
        codes : array-like or None, optional
            N-length array of integers representing the codes of the path.
            If not None, codes must be the same length as vertices.
            If None, *vertices* will be treated as a series of line segments.
        _interpolation_steps : int, optional
            Used as a hint to certain projections, such as Polar, that this
            path should be linearly interpolated immediately before drawing.
            This attribute is primarily an implementation detail and is not
            intended for public use.
        closed : bool, optional
            If *codes* is None and closed is True, vertices will be treated as
            line segments of a closed polygon.  Note that the last vertex will
            then be ignored (as the corresponding code will be set to
            `CLOSEPOLY`).
        readonly : bool, optional
            Makes the path behave in an immutable way and sets the vertices
            and codes as read-only arrays.
        """
        vertices = _to_unmasked_float_array(vertices)
        _api.check_shape((None, 2), vertices=vertices)
        if codes is not None and len(vertices):
            codes = np.asarray(codes, self.code_type)
            if codes.ndim != 1 or len(codes) != len(vertices):
                raise ValueError(f"'codes' must be a 1D list or array with the same length of 'vertices'. Your vertices have shape {vertices.shape} but your codes have shape {codes.shape}")
            if len(codes) and codes[0] != self.MOVETO:
                raise ValueError(f"The first element of 'code' must be equal to 'MOVETO' ({self.MOVETO}).  Your first code is {codes[0]}")
        elif closed and len(vertices):
            codes = np.empty(len(vertices), dtype=self.code_type)
            codes[0] = self.MOVETO
            codes[1:-1] = self.LINETO
            codes[-1] = self.CLOSEPOLY
        self._vertices = vertices
        self._codes = codes
        self._interpolation_steps = _interpolation_steps
        self._update_values()
        if readonly:
            self._vertices.flags.writeable = False
            if self._codes is not None:
                self._codes.flags.writeable = False
            self._readonly = True
        else:
            self._readonly = False

    def _update_values(self):
        self._simplify_threshold = mpl.rcParams['path.simplify_threshold']
        self._should_simplify = self._simplify_threshold > 0 and mpl.rcParams['path.simplify'] and (len(self._vertices) >= 128) and (self._codes is None or np.all(self._codes <= Path.LINETO))

    @property
    def vertices(self):
        """The vertices of the `Path` as an (N, 2) array."""
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        if self._readonly:
            raise AttributeError("Can't set vertices on a readonly Path")
        self._vertices = vertices
        self._update_values()

    @property
    def codes(self):
        """
        The list of codes in the `Path` as a 1D array.

        Each code is one of `STOP`, `MOVETO`, `LINETO`, `CURVE3`, `CURVE4` or
        `CLOSEPOLY`.  For codes that correspond to more than one vertex
        (`CURVE3` and `CURVE4`), that code will be repeated so that the length
        of `vertices` and `codes` is always the same.
        """
        return self._codes

    @codes.setter
    def codes(self, codes):
        if self._readonly:
            raise AttributeError("Can't set codes on a readonly Path")
        self._codes = codes
        self._update_values()

    @classmethod
    def make_compound_path(cls, *args):
        """
        Concatenate a list of `Path`\\s into a single `Path`, removing all `STOP`\\s.
        """
        if not args:
            return Path(np.empty([0, 2], dtype=np.float32))
        vertices = np.concatenate([path.vertices for path in args])
        codes = np.empty(len(vertices), dtype=cls.code_type)
        i = 0
        for path in args:
            size = len(path.vertices)
            if path.codes is None:
                if size:
                    codes[i] = cls.MOVETO
                    codes[i + 1:i + size] = cls.LINETO
            else:
                codes[i:i + size] = path.codes
            i += size
        not_stop_mask = codes != cls.STOP
        return cls(vertices[not_stop_mask], codes[not_stop_mask])

    def _iter_connected_components(self):
        """Return subpaths split at MOVETOs."""
        if self.codes is None:
            yield self
        else:
            idxs = np.append((self.codes == Path.MOVETO).nonzero()[0], len(self.codes))
            for sl in map(slice, idxs, idxs[1:]):
                yield Path._fast_from_codes_and_verts(self.vertices[sl], self.codes[sl], self)

    def interpolated(self, steps):
        """
        Return a new path with each segment divided into *steps* parts.

        Codes other than `LINETO`, `MOVETO`, and `CLOSEPOLY` are not handled correctly.

        Parameters
        ----------
        steps : int
            The number of segments in the new path for each in the original.

        Returns
        -------
        Path
            The interpolated path.
        """
        if steps == 1 or len(self) == 0:
            return self
        if self.codes is not None and self.MOVETO in self.codes[1:]:
            return self.make_compound_path(*(p.interpolated(steps) for p in self._iter_connected_components()))
        if self.codes is not None and self.CLOSEPOLY in self.codes and (not np.all(self.vertices[self.codes == self.CLOSEPOLY] == self.vertices[0])):
            vertices = self.vertices.copy()
            vertices[self.codes == self.CLOSEPOLY] = vertices[0]
        else:
            vertices = self.vertices
        vertices = simple_linear_interpolation(vertices, steps)
        codes = self.codes
        if codes is not None:
            new_codes = np.full((len(codes) - 1) * steps + 1, Path.LINETO, dtype=self.code_type)
            new_codes[0::steps] = codes
        else:
            new_codes = None
        return Path(vertices, new_codes)
    _unit_rectangle = None
    _unit_regular_polygons = WeakValueDictionary()
    _unit_regular_stars = WeakValueDictionary()
    _unit_circle = None
    _unit_circle_righthalf = None
