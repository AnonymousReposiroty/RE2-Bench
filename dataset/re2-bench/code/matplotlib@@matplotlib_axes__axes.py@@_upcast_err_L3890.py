from numbers import Integral, Number, Real
import numpy as np
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
    _AxesBase, _TransformedBoundsLocator, _process_plot_format)
from matplotlib.container import (
    BarContainer, ErrorbarContainer, PieContainer, StemContainer)
from matplotlib.transforms import _ScaledRotation

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

    @_api.make_keyword_only('3.10', 'label')
    @_preprocess_data(replace_names=['y', 'xmin', 'xmax', 'colors'], label_namer='y')
    def hlines(self, y, xmin, xmax, colors=None, linestyles='solid', label='', **kwargs):
        """
        Plot horizontal lines at each *y* from *xmin* to *xmax*.

        Parameters
        ----------
        y : float or array-like
            y-indexes where to plot the lines.

        xmin, xmax : float or array-like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have the same length.

        colors : :mpltype:`color` or list of color , default: :rc:`lines.color`

        linestyles : {'solid', 'dashed', 'dashdot', 'dotted'}, default: 'solid'

        label : str, default: ''

        Returns
        -------
        `~matplotlib.collections.LineCollection`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs :  `~matplotlib.collections.LineCollection` properties.

        See Also
        --------
        vlines : vertical lines
        axhline : horizontal line across the Axes
        """
        xmin, xmax, y = self._process_unit_info([('x', xmin), ('x', xmax), ('y', y)], kwargs)
        if not np.iterable(y):
            y = [y]
        if not np.iterable(xmin):
            xmin = [xmin]
        if not np.iterable(xmax):
            xmax = [xmax]
        y, xmin, xmax = cbook._combine_masks(y, xmin, xmax)
        y = np.ravel(y)
        xmin = np.ravel(xmin)
        xmax = np.ravel(xmax)
        masked_verts = np.ma.empty((len(y), 2, 2))
        masked_verts[:, 0, 0] = xmin
        masked_verts[:, 0, 1] = y
        masked_verts[:, 1, 0] = xmax
        masked_verts[:, 1, 1] = y
        lines = mcoll.LineCollection(masked_verts, colors=colors, linestyles=linestyles, label=label)
        self.add_collection(lines, autolim=False)
        lines._internal_update(kwargs)
        if len(y) > 0:
            updatex = True
            updatey = True
            if self.name == 'rectilinear':
                datalim = lines.get_datalim(self.transData)
                t = lines.get_transform()
                updatex, updatey = t.contains_branch_separately(self.transData)
                minx = np.nanmin(datalim.xmin)
                maxx = np.nanmax(datalim.xmax)
                miny = np.nanmin(datalim.ymin)
                maxy = np.nanmax(datalim.ymax)
            else:
                minx = np.nanmin(masked_verts[..., 0])
                maxx = np.nanmax(masked_verts[..., 0])
                miny = np.nanmin(masked_verts[..., 1])
                maxy = np.nanmax(masked_verts[..., 1])
            corners = ((minx, miny), (maxx, maxy))
            self.update_datalim(corners, updatex, updatey)
            self._request_autoscale_view()
        return lines

    @_api.make_keyword_only('3.10', 'label')
    @_preprocess_data(replace_names=['x', 'ymin', 'ymax', 'colors'], label_namer='x')
    def vlines(self, x, ymin, ymax, colors=None, linestyles='solid', label='', **kwargs):
        """
        Plot vertical lines at each *x* from *ymin* to *ymax*.

        Parameters
        ----------
        x : float or array-like
            x-indexes where to plot the lines.

        ymin, ymax : float or array-like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have the same length.

        colors : :mpltype:`color` or list of color, default: :rc:`lines.color`

        linestyles : {'solid', 'dashed', 'dashdot', 'dotted'}, default: 'solid'

        label : str, default: ''

        Returns
        -------
        `~matplotlib.collections.LineCollection`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs : `~matplotlib.collections.LineCollection` properties.

        See Also
        --------
        hlines : horizontal lines
        axvline : vertical line across the Axes
        """
        x, ymin, ymax = self._process_unit_info([('x', x), ('y', ymin), ('y', ymax)], kwargs)
        if not np.iterable(x):
            x = [x]
        if not np.iterable(ymin):
            ymin = [ymin]
        if not np.iterable(ymax):
            ymax = [ymax]
        x, ymin, ymax = cbook._combine_masks(x, ymin, ymax)
        x = np.ravel(x)
        ymin = np.ravel(ymin)
        ymax = np.ravel(ymax)
        masked_verts = np.ma.empty((len(x), 2, 2))
        masked_verts[:, 0, 0] = x
        masked_verts[:, 0, 1] = ymin
        masked_verts[:, 1, 0] = x
        masked_verts[:, 1, 1] = ymax
        lines = mcoll.LineCollection(masked_verts, colors=colors, linestyles=linestyles, label=label)
        self.add_collection(lines, autolim=False)
        lines._internal_update(kwargs)
        if len(x) > 0:
            updatex = True
            updatey = True
            if self.name == 'rectilinear':
                datalim = lines.get_datalim(self.transData)
                t = lines.get_transform()
                updatex, updatey = t.contains_branch_separately(self.transData)
                minx = np.nanmin(datalim.xmin)
                maxx = np.nanmax(datalim.xmax)
                miny = np.nanmin(datalim.ymin)
                maxy = np.nanmax(datalim.ymax)
            else:
                minx = np.nanmin(masked_verts[..., 0])
                maxx = np.nanmax(masked_verts[..., 0])
                miny = np.nanmin(masked_verts[..., 1])
                maxy = np.nanmax(masked_verts[..., 1])
            corners = ((minx, miny), (maxx, maxy))
            self.update_datalim(corners, updatex, updatey)
            self._request_autoscale_view()
        return lines

    @staticmethod
    def _errorevery_to_mask(x, errorevery):
        """
        Normalize `errorbar`'s *errorevery* to be a boolean mask for data *x*.

        This function is split out to be usable both by 2D and 3D errorbars.
        """
        if isinstance(errorevery, Integral):
            errorevery = (0, errorevery)
        if isinstance(errorevery, tuple):
            if len(errorevery) == 2 and isinstance(errorevery[0], Integral) and isinstance(errorevery[1], Integral):
                errorevery = slice(errorevery[0], None, errorevery[1])
            else:
                raise ValueError(f'errorevery={errorevery!r} is a not a tuple of two integers')
        elif isinstance(errorevery, slice):
            pass
        elif not isinstance(errorevery, str) and np.iterable(errorevery):
            try:
                x[errorevery]
            except (ValueError, IndexError) as err:
                raise ValueError(f"errorevery={errorevery!r} is iterable but not a valid NumPy fancy index to match 'xerr'/'yerr'") from err
        else:
            raise ValueError(f'errorevery={errorevery!r} is not a recognized value')
        everymask = np.zeros(len(x), bool)
        everymask[errorevery] = True
        return everymask

    @_api.make_keyword_only('3.10', 'ecolor')
    @_preprocess_data(replace_names=['x', 'y', 'xerr', 'yerr'], label_namer='y')
    @_docstring.interpd
    def errorbar(self, x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, elinestyle=None, **kwargs):
        """
        Plot y versus x as lines and/or markers with attached errorbars.

        *x*, *y* define the data locations, *xerr*, *yerr* define the errorbar
        sizes. By default, this draws the data markers/lines as well as the
        errorbars. Use fmt='none' to draw errorbars without any data markers.

        .. versionadded:: 3.7
           Caps and error lines are drawn in polar coordinates on polar plots.


        Parameters
        ----------
        x, y : float or array-like
            The data positions.

        xerr, yerr : float or array-like, shape(N,) or shape(2, N), optional
            The errorbar sizes:

            - scalar: Symmetric +/- values for all data points.
            - shape(N,): Symmetric +/-values for each data point.
            - shape(2, N): Separate - and + values for each bar. First row
              contains the lower errors, the second row contains the upper
              errors.
            - *None*: No errorbar.

            All values must be >= 0.

            See :doc:`/gallery/statistics/errorbar_features`
            for an example on the usage of ``xerr`` and ``yerr``.

        fmt : str, default: ''
            The format for the data points / data lines. See `.plot` for
            details.

            Use 'none' (case-insensitive) to plot errorbars without any data
            markers.

        ecolor : :mpltype:`color`, default: None
            The color of the errorbar lines.  If None, use the color of the
            line connecting the markers.

        elinewidth : float, default: None
            The linewidth of the errorbar lines. If None, the linewidth of
            the current style is used.

        elinestyle : str or tuple, default: 'solid'
           The linestyle of the errorbar lines.
           Valid values for linestyles include {'-', '--', '-.',
            ':', '', (offset, on-off-seq)}. See `.Line2D.set_linestyle` for a
            complete description.

        capsize : float, default: :rc:`errorbar.capsize`
            The length of the error bar caps in points.

        capthick : float, default: None
            An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).
            This setting is a more sensible name for the property that
            controls the thickness of the error bar cap in points. For
            backwards compatibility, if *mew* or *markeredgewidth* are given,
            then they will over-ride *capthick*. This may change in future
            releases.

        barsabove : bool, default: False
            If True, will plot the errorbars above the plot
            symbols. Default is below.

        lolims, uplims, xlolims, xuplims : bool or array-like, default: False
            These arguments can be used to indicate that a value gives only
            upper/lower limits.  In that case a caret symbol is used to
            indicate this. *lims*-arguments may be scalars, or array-likes of
            the same length as *xerr* and *yerr*.  To use limits with inverted
            axes, `~.Axes.set_xlim` or `~.Axes.set_ylim` must be called before
            :meth:`errorbar`.  Note the tricky parameter names: setting e.g.
            *lolims* to True means that the y-value is a *lower* limit of the
            True value, so, only an *upward*-pointing arrow will be drawn!

        errorevery : int or (int, int), default: 1
            draws error bars on a subset of the data. *errorevery* =N draws
            error bars on the points (x[::N], y[::N]).
            *errorevery* =(start, N) draws error bars on the points
            (x[start::N], y[start::N]). e.g. errorevery=(6, 3)
            adds error bars to the data at (x[6], x[9], x[12], x[15], ...).
            Used to avoid overlapping error bars when two series share x-axis
            values.

        Returns
        -------
        `.ErrorbarContainer`
            The container contains:

            - data_line : A `~matplotlib.lines.Line2D` instance of x, y plot markers
              and/or line.
            - caplines : A tuple of `~matplotlib.lines.Line2D` instances of the error
              bar caps.
            - barlinecols : A tuple of `.LineCollection` with the horizontal and
              vertical error ranges.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            All other keyword arguments are passed on to the `~.Axes.plot` call
            drawing the markers. For example, this code makes big red squares
            with thick green edges::

                x, y, yerr = rand(3, 10)
                errorbar(x, y, yerr, marker='s', mfc='red',
                         mec='green', ms=20, mew=4)

            where *mfc*, *mec*, *ms* and *mew* are aliases for the longer
            property names, *markerfacecolor*, *markeredgecolor*, *markersize*
            and *markeredgewidth*.

            Valid kwargs for the marker properties are:

            - *dashes*
            - *dash_capstyle*
            - *dash_joinstyle*
            - *drawstyle*
            - *fillstyle*
            - *linestyle*
            - *marker*
            - *markeredgecolor*
            - *markeredgewidth*
            - *markerfacecolor*
            - *markerfacecoloralt*
            - *markersize*
            - *markevery*
            - *solid_capstyle*
            - *solid_joinstyle*

            Refer to the corresponding `.Line2D` property for more details:

            %(Line2D:kwdoc)s
        """
        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.setdefault('zorder', 2)
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=object)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=object)

        def _upcast_err(err):
            """
            Safely handle tuple of containers that carry units.

            This function covers the case where the input to the xerr/yerr is a
            length 2 tuple of equal length ndarray-subclasses that carry the
            unit information in the container.

            If we have a tuple of nested numpy array (subclasses), we defer
            coercing the units to be consistent to the underlying unit
            library (and implicitly the broadcasting).

            Otherwise, fallback to casting to an object array.
            """
            if np.iterable(err) and len(err) > 0 and isinstance(cbook._safe_first_finite(err), np.ndarray):
                atype = type(cbook._safe_first_finite(err))
                if atype is np.ndarray:
                    return np.asarray(err, dtype=object)
                return atype(err)
            return np.asarray(err, dtype=object)
        if xerr is not None and (not isinstance(xerr, np.ndarray)):
            xerr = _upcast_err(xerr)
        if yerr is not None and (not isinstance(yerr, np.ndarray)):
            yerr = _upcast_err(yerr)
        x, y = np.atleast_1d(x, y)
        if len(x) != len(y):
            raise ValueError("'x' and 'y' must have the same size")
        everymask = self._errorevery_to_mask(x, errorevery)
        label = kwargs.pop('label', None)
        kwargs['label'] = '_nolegend_'
        (data_line, base_style), = self._get_lines._plot_args(self, (x, y) if fmt == '' else (x, y, fmt), kwargs, return_kwargs=True)
        if barsabove:
            data_line.set_zorder(kwargs['zorder'] - 0.1)
        else:
            data_line.set_zorder(kwargs['zorder'] + 0.1)
        if fmt.lower() != 'none':
            self.add_line(data_line)
        else:
            data_line = None
            base_style.pop('color')
            if 'color' in kwargs:
                base_style['color'] = kwargs.pop('color')
        if 'color' not in base_style:
            base_style['color'] = 'C0'
        if ecolor is None:
            ecolor = base_style['color']
        for key in ['marker', 'markersize', 'markerfacecolor', 'markerfacecoloralt', 'markeredgewidth', 'markeredgecolor', 'markevery', 'linestyle', 'fillstyle', 'drawstyle', 'dash_capstyle', 'dash_joinstyle', 'solid_capstyle', 'solid_joinstyle', 'dashes']:
            base_style.pop(key, None)
        eb_lines_style = {**base_style, 'color': ecolor}
        if elinewidth is not None:
            eb_lines_style['linewidth'] = elinewidth
        elif 'linewidth' in kwargs:
            eb_lines_style['linewidth'] = kwargs['linewidth']
        for key in ('transform', 'alpha', 'zorder', 'rasterized'):
            if key in kwargs:
                eb_lines_style[key] = kwargs[key]
        if elinestyle is not None:
            eb_lines_style['linestyle'] = elinestyle
        eb_cap_style = {**base_style, 'linestyle': 'none'}
        capsize = mpl._val_or_rc(capsize, 'errorbar.capsize')
        if capsize > 0:
            eb_cap_style['markersize'] = 2.0 * capsize
        if capthick is not None:
            eb_cap_style['markeredgewidth'] = capthick
        for key in ('markeredgewidth', 'transform', 'alpha', 'zorder', 'rasterized'):
            if key in kwargs:
                eb_cap_style[key] = kwargs[key]
        eb_cap_style['markeredgecolor'] = ecolor
        barcols = []
        caplines = {'x': [], 'y': []}

        def apply_mask(arrays, mask):
            return [array[mask] for array in arrays]
        for dep_axis, dep, err, lolims, uplims, indep, lines_func, marker, lomarker, himarker in [('x', x, xerr, xlolims, xuplims, y, self.hlines, '|', mlines.CARETRIGHTBASE, mlines.CARETLEFTBASE), ('y', y, yerr, lolims, uplims, x, self.vlines, '_', mlines.CARETUPBASE, mlines.CARETDOWNBASE)]:
            if err is None:
                continue
            lolims = np.broadcast_to(lolims, len(dep)).astype(bool)
            uplims = np.broadcast_to(uplims, len(dep)).astype(bool)
            try:
                np.broadcast_to(err, (2, len(dep)))
            except ValueError:
                raise ValueError(f"'{dep_axis}err' (shape: {np.shape(err)}) must be a scalar or a 1D or (2, n) array-like whose shape matches '{dep_axis}' (shape: {np.shape(dep)})") from None
            if err.dtype is np.dtype(object) and np.any(err == None):
                raise ValueError(f"'{dep_axis}err' must not contain None. Use NaN if you want to skip a value.")
            if np.any((check := err[err == err]) < -check):
                raise ValueError(f"'{dep_axis}err' must not contain negative values")
            low, high = dep + np.vstack([-(1 - lolims), 1 - uplims]) * err
            barcols.append(lines_func(*apply_mask([indep, low, high], everymask), **eb_lines_style))
            if self.name == 'polar' and dep_axis == 'x':
                for b in barcols:
                    for p in b.get_paths():
                        p._interpolation_steps = 2
            nolims = ~(lolims | uplims)
            if nolims.any() and capsize > 0:
                indep_masked, lo_masked, hi_masked = apply_mask([indep, low, high], nolims & everymask)
                for lh_masked in [lo_masked, hi_masked]:
                    line = mlines.Line2D(indep_masked, indep_masked, marker=marker, **eb_cap_style)
                    line.set(**{f'{dep_axis}data': lh_masked})
                    caplines[dep_axis].append(line)
            for idx, (lims, hl) in enumerate([(lolims, high), (uplims, low)]):
                if not lims.any():
                    continue
                hlmarker = himarker if self._axis_map[dep_axis].get_inverted() ^ idx else lomarker
                x_masked, y_masked, hl_masked = apply_mask([x, y, hl], lims & everymask)
                line = mlines.Line2D(x_masked, y_masked, marker=hlmarker, **eb_cap_style)
                line.set(**{f'{dep_axis}data': hl_masked})
                caplines[dep_axis].append(line)
                if capsize > 0:
                    caplines[dep_axis].append(mlines.Line2D(x_masked, y_masked, marker=marker, **eb_cap_style))
        if self.name == 'polar':
            trans_shift = self.transShift
            for axis in caplines:
                for l in caplines[axis]:
                    for theta, r in zip(l.get_xdata(), l.get_ydata()):
                        rotation = _ScaledRotation(theta=theta, trans_shift=trans_shift)
                        if axis == 'y':
                            rotation += mtransforms.Affine2D().rotate(np.pi / 2)
                        ms = mmarkers.MarkerStyle(marker=marker, transform=rotation)
                        self.add_line(mlines.Line2D([theta], [r], marker=ms, **eb_cap_style))
        else:
            for axis in caplines:
                for l in caplines[axis]:
                    self.add_line(l)
        self._request_autoscale_view()
        caplines = caplines['x'] + caplines['y']
        errorbar_container = ErrorbarContainer((data_line, tuple(caplines), tuple(barcols)), has_xerr=xerr is not None, has_yerr=yerr is not None, label=label)
        self.add_container(errorbar_container)
        return errorbar_container
    if _fill_between_x_or_y.__doc__:
        fill_between.__doc__ = _fill_between_x_or_y.__doc__.format(dir='horizontal', ind='x', dep='y')
    fill_between = _preprocess_data(_docstring.interpd(fill_between), replace_names=['x', 'y1', 'y2', 'where'])
    if _fill_between_x_or_y.__doc__:
        fill_betweenx.__doc__ = _fill_between_x_or_y.__doc__.format(dir='vertical', ind='y', dep='x')
    fill_betweenx = _preprocess_data(_docstring.interpd(fill_betweenx), replace_names=['y', 'x1', 'x2', 'where'])
    table = _make_axes_method(mtable.table)
    stackplot = _preprocess_data()(_make_axes_method(mstack.stackplot))
    streamplot = _preprocess_data(replace_names=['x', 'y', 'u', 'v', 'start_points'])(_make_axes_method(mstream.streamplot))
    tricontour = _make_axes_method(mtri.tricontour)
    tricontourf = _make_axes_method(mtri.tricontourf)
    tripcolor = _make_axes_method(mtri.tripcolor)
    triplot = _make_axes_method(mtri.triplot)

    def _request_autoscale_view(self, axis='all', tight=None):
        """
        Mark a single axis, or all of them, as stale wrt. autoscaling.

        No computation is performed until the next autoscaling; thus, separate
        calls to control individual `Axis`s incur negligible performance cost.

        Parameters
        ----------
        axis : str, default: "all"
            Either an element of ``self._axis_names``, or "all".
        tight : bool or None, default: None
        """
        axis_names = _api.check_getitem({**{k: [k] for k in self._axis_names}, 'all': self._axis_names}, axis=axis)
        for name in axis_names:
            self._stale_viewlims[name] = True
        if tight is not None:
            self._tight = tight

    def add_collection(self, collection, autolim=True):
        """
        Add a `.Collection` to the Axes; return the collection.

        Parameters
        ----------
        collection : `.Collection`
            The collection to add.
        autolim : bool
            Whether to update data and view limits.

            .. versionchanged:: 3.11

               This now also updates the view limits, making explicit
               calls to `~.Axes.autoscale_view` unnecessary.

            As an implementation detail, the value "_datalim_only" is
            supported to smooth the internal transition from pre-3.11
            behavior. This is not a public interface and will be removed
            again in the future.
        """
        _api.check_isinstance(mcoll.Collection, collection=collection)
        if not collection.get_label():
            collection.set_label(f'_child{len(self._children)}')
        self._children.append(collection)
        collection._remove_method = self._children.remove
        self._set_artist_props(collection)
        if collection.get_clip_path() is None:
            collection.set_clip_path(self.patch)
        if autolim:
            self._unstale_viewLim()
            datalim = collection.get_datalim(self.transData)
            points = datalim.get_points()
            if not np.isinf(datalim.minpos).all():
                points = np.concatenate([points, [datalim.minpos]])
            x_is_data, y_is_data = collection.get_transform().contains_branch_separately(self.transData)
            ox_is_data, oy_is_data = collection.get_offset_transform().contains_branch_separately(self.transData)
            self.update_datalim(points, updatex=x_is_data or ox_is_data, updatey=y_is_data or oy_is_data)
            if autolim != '_datalim_only':
                self._request_autoscale_view()
        self.stale = True
        return collection

    def add_line(self, line):
        """
        Add a `.Line2D` to the Axes; return the line.
        """
        _api.check_isinstance(mlines.Line2D, line=line)
        self._set_artist_props(line)
        if line.get_clip_path() is None:
            line.set_clip_path(self.patch)
        self._update_line_limits(line)
        if not line.get_label():
            line.set_label(f'_child{len(self._children)}')
        self._children.append(line)
        line._remove_method = self._children.remove
        self.stale = True
        return line

    def add_container(self, container):
        """
        Add a `.Container` to the Axes' containers; return the container.
        """
        label = container.get_label()
        if not label:
            container.set_label('_container%d' % len(self.containers))
        self.containers.append(container)
        container._remove_method = self.containers.remove
        return container

    def update_datalim(self, xys, updatex=True, updatey=True):
        """
        Extend the `~.Axes.dataLim` Bbox to include the given points.

        If no data is set currently, the Bbox will ignore its limits and set
        the bound to be the bounds of the xydata (*xys*). Otherwise, it will
        compute the bounds of the union of its current data and the data in
        *xys*.

        Parameters
        ----------
        xys : 2D array-like
            The points to include in the data limits Bbox. This can be either
            a list of (x, y) tuples or a (N, 2) array.

        updatex, updatey : bool, default: True
            Whether to update the x/y limits.
        """
        xys = np.asarray(xys)
        if not np.any(np.isfinite(xys)):
            return
        self.dataLim.update_from_data_xy(xys, self.ignore_existing_data_limits, updatex=updatex, updatey=updatey)
        self.ignore_existing_data_limits = False

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
