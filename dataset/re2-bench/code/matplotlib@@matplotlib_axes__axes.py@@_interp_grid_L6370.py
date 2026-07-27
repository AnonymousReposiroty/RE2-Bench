import numpy as np
import matplotlib.cbook as cbook
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.tri as mtri
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
    _AxesBase, _TransformedBoundsLocator, _process_plot_format)

@_docstring.interpd
class Axes(_AxesBase):
    """
    An Axes object encapsulates all the elements of an individual (sub-)plot in
    a figure.

    It contains most of the (sub-)plot elements: `~.axis.Axis`,
    `~.axis.Tick`, `~.lines.Line2D`, `~.text.Text`, `~.patches.Polygon`, etc.,
    and sets the coordinate system.

    Like all visible elements in a figure, Axes is an `.Artist` subclass.

    The `Axes` instance supports callbacks through a callbacks attribute which
    is a `~.cbook.CallbackRegistry` instance.  The events you can connect to
    are 'xlim_changed' and 'ylim_changed' and the callback will be called with
    func(*ax*) where *ax* is the `Axes` instance.

    .. note::

        As a user, you do not instantiate Axes directly, but use Axes creation
        methods instead; e.g. from `.pyplot` or `.Figure`:
        `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.

    """
    annotate.__doc__ = mtext.Annotation.__init__.__doc__
    if _fill_between_x_or_y.__doc__:
        fill_between.__doc__ = _fill_between_x_or_y.__doc__.format(dir='horizontal', ind='x', dep='y')
    fill_between = _preprocess_data(_docstring.interpd(fill_between), replace_names=['x', 'y1', 'y2', 'where'])
    if _fill_between_x_or_y.__doc__:
        fill_betweenx.__doc__ = _fill_between_x_or_y.__doc__.format(dir='vertical', ind='y', dep='x')
    fill_betweenx = _preprocess_data(_docstring.interpd(fill_betweenx), replace_names=['y', 'x1', 'x2', 'where'])

    def _pcolorargs(self, funcname, *args, shading='auto', **kwargs):
        _valid_shading = ['gouraud', 'nearest', 'flat', 'auto']
        try:
            _api.check_in_list(_valid_shading, shading=shading)
        except ValueError:
            _api.warn_external(f"shading value '{shading}' not in list of valid values {_valid_shading}. Setting shading='auto'.")
            shading = 'auto'
        if len(args) == 1:
            C = np.asanyarray(args[0])
            nrows, ncols = C.shape[:2]
            if shading in ['gouraud', 'nearest']:
                X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
            else:
                X, Y = np.meshgrid(np.arange(ncols + 1), np.arange(nrows + 1))
                shading = 'flat'
        elif len(args) == 3:
            C = np.asanyarray(args[2])
            X, Y = args[:2]
            X, Y = self._process_unit_info([('x', X), ('y', Y)], kwargs)
            X, Y = (cbook.safe_masked_invalid(a, copy=True) for a in [X, Y])
            if funcname == 'pcolormesh':
                if np.ma.is_masked(X) or np.ma.is_masked(Y):
                    raise ValueError('x and y arguments to pcolormesh cannot have non-finite values or be of type numpy.ma.MaskedArray with masked values')
            nrows, ncols = C.shape[:2]
        else:
            raise _api.nargs_error(funcname, takes='1 or 3', given=len(args))
        Nx = X.shape[-1]
        Ny = Y.shape[0]
        if X.ndim != 2 or X.shape[0] == 1:
            x = X.reshape(1, Nx)
            X = x.repeat(Ny, axis=0)
        if Y.ndim != 2 or Y.shape[1] == 1:
            y = Y.reshape(Ny, 1)
            Y = y.repeat(Nx, axis=1)
        if X.shape != Y.shape:
            raise TypeError(f'Incompatible X, Y inputs to {funcname}; see help({funcname})')
        if shading == 'auto':
            if ncols == Nx and nrows == Ny:
                shading = 'nearest'
            else:
                shading = 'flat'
        if shading == 'flat':
            if (Nx, Ny) != (ncols + 1, nrows + 1):
                raise TypeError(f"Dimensions of C {C.shape} should be one smaller than X({Nx}) and Y({Ny}) while using shading='flat' see help({funcname})")
        else:
            if (Nx, Ny) != (ncols, nrows):
                raise TypeError('Dimensions of C %s are incompatible with X (%d) and/or Y (%d); see help(%s)' % (C.shape, Nx, Ny, funcname))
            if shading == 'nearest':

                def _interp_grid(X, require_monotonicity=False):
                    if np.shape(X)[1] > 1:
                        dX = np.diff(X, axis=1) * 0.5
                        if require_monotonicity and (not (np.all(dX >= 0) or np.all(dX <= 0))):
                            _api.warn_external(f'The input coordinates to {funcname} are interpreted as cell centers, but are not monotonically increasing or decreasing. This may lead to incorrectly calculated cell edges, in which case, please supply explicit cell edges to {funcname}.')
                        hstack = np.ma.hstack if np.ma.isMA(X) else np.hstack
                        X = hstack((X[:, [0]] - dX[:, [0]], X[:, :-1] + dX, X[:, [-1]] + dX[:, [-1]]))
                    else:
                        X = np.hstack((X, X))
                    return X
                if ncols == Nx:
                    X = _interp_grid(X, require_monotonicity=True)
                    Y = _interp_grid(Y)
                if nrows == Ny:
                    X = _interp_grid(X.T).T
                    Y = _interp_grid(Y.T, require_monotonicity=True).T
                shading = 'flat'
        C = cbook.safe_masked_invalid(C, copy=True)
        return (X, Y, C, shading)
    table = _make_axes_method(mtable.table)
    stackplot = _preprocess_data()(_make_axes_method(mstack.stackplot))
    streamplot = _preprocess_data(replace_names=['x', 'y', 'u', 'v', 'start_points'])(_make_axes_method(mstream.streamplot))
    tricontour = _make_axes_method(mtri.tricontour)
    tricontourf = _make_axes_method(mtri.tricontourf)
    tripcolor = _make_axes_method(mtri.tripcolor)
    triplot = _make_axes_method(mtri.triplot)

    def _process_unit_info(self, datasets=None, kwargs=None, *, convert=True):
        """
        Set axis units based on *datasets* and *kwargs*, and optionally apply
        unit conversions to *datasets*.

        Parameters
        ----------
        datasets : list
            List of (axis_name, dataset) pairs (where the axis name is defined
            as in `._axis_map`).  Individual datasets can also be None
            (which gets passed through).
        kwargs : dict
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D Axes), *runits* and *thetaunits* (for
            polar) entries) is popped, if present.  Note that this dict is
            mutated in-place!
        convert : bool, default: True
            Whether to return the original datasets or the converted ones.

        Returns
        -------
        list
            Either the original datasets if *convert* is False, or the
            converted ones if *convert* is True (the default).
        """
        datasets = datasets or []
        kwargs = kwargs or {}
        axis_map = self._axis_map
        for axis_name, data in datasets:
            try:
                axis = axis_map[axis_name]
            except KeyError:
                raise ValueError(f'Invalid axis name: {axis_name!r}') from None
            if axis is not None and data is not None and (not axis.have_units()):
                axis.update_units(data)
        for axis_name, axis in axis_map.items():
            if axis is None:
                continue
            units = kwargs.pop(f'{axis_name}units', axis.units)
            if self.name == 'polar':
                polar_units = {'x': 'thetaunits', 'y': 'runits'}
                units = kwargs.pop(polar_units[axis_name], units)
            if units != axis.units and units is not None:
                axis.set_units(units)
                for dataset_axis_name, data in datasets:
                    if dataset_axis_name == axis_name and data is not None:
                        axis.update_units(data)
        return [axis_map[axis_name].convert_units(data) if convert and data is not None else data for axis_name, data in datasets]
