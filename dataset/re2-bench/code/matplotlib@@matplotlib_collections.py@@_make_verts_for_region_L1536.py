import functools
import numpy as np
from . import (_api, _path, artist, cbook, colorizer as mcolorizer, colors as mcolors,
               _docstring, hatch as mhatch, lines as mlines, path as mpath, transforms)

class FillBetweenPolyCollection(PolyCollection):
    """
    `.PolyCollection` that fills the area between two x- or y-curves.
    """

    def __init__(self, t_direction, t, f1, f2, *, where=None, interpolate=False, step=None, **kwargs):
        """
        Parameters
        ----------
        t_direction : {{'x', 'y'}}
            The axes on which the variable lies.

            - 'x': the curves are ``(t, f1)`` and ``(t, f2)``.
            - 'y': the curves are ``(f1, t)`` and ``(f2, t)``.

        t : array-like
            The ``t_direction`` coordinates of the nodes defining the curves.

        f1 : array-like or float
            The other coordinates of the nodes defining the first curve.

        f2 : array-like or float
            The other coordinates of the nodes defining the second curve.

        where : array-like of bool, optional
            Define *where* to exclude some {dir} regions from being filled.
            The filled regions are defined by the coordinates ``t[where]``.
            More precisely, fill between ``t[i]`` and ``t[i+1]`` if
            ``where[i] and where[i+1]``.  Note that this definition implies
            that an isolated *True* value between two *False* values in *where*
            will not result in filling.  Both sides of the *True* position
            remain unfilled due to the adjacent *False* values.

        interpolate : bool, default: False
            This option is only relevant if *where* is used and the two curves
            are crossing each other.

            Semantically, *where* is often used for *f1* > *f2* or
            similar.  By default, the nodes of the polygon defining the filled
            region will only be placed at the positions in the *t* array.
            Such a polygon cannot describe the above semantics close to the
            intersection.  The t-sections containing the intersection are
            simply clipped.

            Setting *interpolate* to *True* will calculate the actual
            intersection point and extend the filled region up to this point.

        step : {{'pre', 'post', 'mid'}}, optional
            Define *step* if the filling should be a step function,
            i.e. constant in between *t*.  The value determines where the
            step will occur:

            - 'pre': The f value is continued constantly to the left from
              every *t* position, i.e. the interval ``(t[i-1], t[i]]`` has the
              value ``f[i]``.
            - 'post': The y value is continued constantly to the right from
              every *x* position, i.e. the interval ``[t[i], t[i+1])`` has the
              value ``f[i]``.
            - 'mid': Steps occur half-way between the *t* positions.

        **kwargs
            Forwarded to `.PolyCollection`.

        See Also
        --------
        .Axes.fill_between, .Axes.fill_betweenx
        """
        self.t_direction = t_direction
        self._interpolate = interpolate
        self._step = step
        verts = self._make_verts(t, f1, f2, where)
        super().__init__(verts, **kwargs)

    @staticmethod
    def _f_dir_from_t(t_direction):
        """The direction that is other than `t_direction`."""
        if t_direction == 'x':
            return 'y'
        elif t_direction == 'y':
            return 'x'
        else:
            msg = f"t_direction must be 'x' or 'y', got {t_direction!r}"
            raise ValueError(msg)

    @property
    def _f_direction(self):
        """The direction that is other than `self.t_direction`."""
        return self._f_dir_from_t(self.t_direction)

    def _make_verts(self, t, f1, f2, where):
        """
        Make verts that can be forwarded to `.PolyCollection`.
        """
        self._validate_shapes(self.t_direction, self._f_direction, t, f1, f2)
        where = self._get_data_mask(t, f1, f2, where)
        t, f1, f2 = np.broadcast_arrays(np.atleast_1d(t), f1, f2, subok=True)
        self._bbox = transforms.Bbox.null()
        self._bbox.update_from_data_xy(self._fix_pts_xy_order(np.concatenate([np.stack((t[where], f[where]), axis=-1) for f in (f1, f2)])))
        return [self._make_verts_for_region(t, f1, f2, idx0, idx1) for idx0, idx1 in cbook.contiguous_regions(where)]

    def _get_data_mask(self, t, f1, f2, where):
        """
        Return a bool array, with True at all points that should eventually be rendered.

        The array is True at a point if none of the data inputs
        *t*, *f1*, *f2* is masked and if the input *where* is true at that point.
        """
        if where is None:
            where = True
        else:
            where = np.asarray(where, dtype=bool)
            if where.size != t.size:
                msg = 'where size ({}) does not match {!r} size ({})'.format(where.size, self.t_direction, t.size)
                raise ValueError(msg)
        return where & ~functools.reduce(np.logical_or, map(np.ma.getmaskarray, [t, f1, f2]))

    @staticmethod
    def _validate_shapes(t_dir, f_dir, t, f1, f2):
        """Validate that t, f1 and f2 are 1-dimensional and have the same length."""
        names = (d + s for d, s in zip((t_dir, f_dir, f_dir), ('', '1', '2')))
        for name, array in zip(names, [t, f1, f2]):
            if array.ndim > 1:
                raise ValueError(f'{name!r} is not 1-dimensional')
            if t.size > 1 and array.size > 1 and (t.size != array.size):
                msg = '{!r} has size {}, but {!r} has an unequal size of {}'.format(t_dir, t.size, name, array.size)
                raise ValueError(msg)

    def _make_verts_for_region(self, t, f1, f2, idx0, idx1):
        """
        Make ``verts`` for a contiguous region between ``idx0`` and ``idx1``, taking
        into account ``step`` and ``interpolate``.
        """
        t_slice = t[idx0:idx1]
        f1_slice = f1[idx0:idx1]
        f2_slice = f2[idx0:idx1]
        if self._step is not None:
            step_func = cbook.STEP_LOOKUP_MAP['steps-' + self._step]
            t_slice, f1_slice, f2_slice = step_func(t_slice, f1_slice, f2_slice)
        if self._interpolate:
            start = self._get_interpolating_points(t, f1, f2, idx0)
            end = self._get_interpolating_points(t, f1, f2, idx1)
        else:
            start = (t_slice[0], f2_slice[0])
            end = (t_slice[-1], f2_slice[-1])
        pts = np.concatenate((np.asarray([start]), np.stack((t_slice, f1_slice), axis=-1), np.asarray([end]), np.stack((t_slice, f2_slice), axis=-1)[::-1]))
        return self._fix_pts_xy_order(pts)

    @classmethod
    def _get_interpolating_points(cls, t, f1, f2, idx):
        """Calculate interpolating points."""
        im1 = max(idx - 1, 0)
        t_values = t[im1:idx + 1]
        diff_values = f1[im1:idx + 1] - f2[im1:idx + 1]
        f1_values = f1[im1:idx + 1]
        if len(diff_values) == 2:
            if np.ma.is_masked(diff_values[1]):
                return (t[im1], f1[im1])
            elif np.ma.is_masked(diff_values[0]):
                return (t[idx], f1[idx])
        diff_root_t = cls._get_diff_root(0, diff_values, t_values)
        diff_root_f = cls._get_diff_root(diff_root_t, t_values, f1_values)
        return (diff_root_t, diff_root_f)

    @staticmethod
    def _get_diff_root(x, xp, fp):
        """Calculate diff root."""
        order = xp.argsort()
        return np.interp(x, xp[order], fp[order])

    def _fix_pts_xy_order(self, pts):
        """
        Fix pts calculation results with `self.t_direction`.

        In the workflow, it is assumed that `self.t_direction` is 'x'. If this
        is not true, we need to exchange the coordinates.
        """
        return pts[:, ::-1] if self.t_direction == 'y' else pts
