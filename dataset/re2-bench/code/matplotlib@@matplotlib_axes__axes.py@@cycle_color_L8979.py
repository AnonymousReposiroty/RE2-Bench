import functools
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.tri as mtri
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
    _AxesBase, _TransformedBoundsLocator, _process_plot_format)
from builtins import range

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

    def _fill_between_x_or_y(self, ind_dir, ind, dep1, dep2=0, *, where=None, interpolate=False, step=None, **kwargs):
        """
        Fill the area between two {dir} curves.

        The curves are defined by the points (*{ind}*, *{dep}1*) and (*{ind}*,
        *{dep}2*).  This creates one or multiple polygons describing the filled
        area.

        You may exclude some {dir} sections from filling using *where*.

        By default, the edges connect the given points directly.  Use *step*
        if the filling should be a step function, i.e. constant in between
        *{ind}*.

        Parameters
        ----------
        {ind} : array-like
            The {ind} coordinates of the nodes defining the curves.

        {dep}1 : array-like or float
            The {dep} coordinates of the nodes defining the first curve.

        {dep}2 : array-like or float, default: 0
            The {dep} coordinates of the nodes defining the second curve.

        where : array-like of bool, optional
            Define *where* to exclude some {dir} regions from being filled.
            The filled regions are defined by the coordinates ``{ind}[where]``.
            More precisely, fill between ``{ind}[i]`` and ``{ind}[i+1]`` if
            ``where[i] and where[i+1]``.  Note that this definition implies
            that an isolated *True* value between two *False* values in *where*
            will not result in filling.  Both sides of the *True* position
            remain unfilled due to the adjacent *False* values.

        interpolate : bool, default: False
            This option is only relevant if *where* is used and the two curves
            are crossing each other.

            Semantically, *where* is often used for *{dep}1* > *{dep}2* or
            similar.  By default, the nodes of the polygon defining the filled
            region will only be placed at the positions in the *{ind}* array.
            Such a polygon cannot describe the above semantics close to the
            intersection.  The {ind}-sections containing the intersection are
            simply clipped.

            Setting *interpolate* to *True* will calculate the actual
            intersection point and extend the filled region up to this point.

        step : {{'pre', 'post', 'mid'}}, optional
            Define *step* if the filling should be a step function,
            i.e. constant in between *{ind}*.  The value determines where the
            step will occur:

            - 'pre': The {dep} value is continued constantly to the left from
              every *{ind}* position, i.e. the interval ``({ind}[i-1], {ind}[i]]``
              has the value ``{dep}[i]``.
            - 'post': The y value is continued constantly to the right from
              every *{ind}* position, i.e. the interval ``[{ind}[i], {ind}[i+1])``
              has the value ``{dep}[i]``.
            - 'mid': Steps occur half-way between the *{ind}* positions.

        Returns
        -------
        `.FillBetweenPolyCollection`
            A `.FillBetweenPolyCollection` containing the plotted polygons.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            All other keyword arguments are passed on to
            `.FillBetweenPolyCollection`. They control the `.Polygon` properties:

            %(FillBetweenPolyCollection:kwdoc)s

        See Also
        --------
        fill_between : Fill between two sets of y-values.
        fill_betweenx : Fill between two sets of x-values.
        """
        dep_dir = mcoll.FillBetweenPolyCollection._f_dir_from_t(ind_dir)
        if not mpl.rcParams['_internal.classic_mode']:
            kwargs = cbook.normalize_kwargs(kwargs, mcoll.Collection)
            if not any((c in kwargs for c in ('color', 'facecolor'))):
                kwargs['facecolor'] = self._get_patches_for_fill.get_next_color()
        ind, dep1, dep2 = self._fill_between_process_units(ind_dir, dep_dir, ind, dep1, dep2, **kwargs)
        collection = mcoll.FillBetweenPolyCollection(ind_dir, ind, dep1, dep2, where=where, interpolate=interpolate, step=step, **kwargs)
        self.add_collection(collection)
        return collection

    def _fill_between_process_units(self, ind_dir, dep_dir, ind, dep1, dep2, **kwargs):
        """Handle united data, such as dates."""
        return map(np.ma.masked_invalid, self._process_unit_info([(ind_dir, ind), (dep_dir, dep1), (dep_dir, dep2)], kwargs))

    def fill_between(self, x, y1, y2=0, where=None, interpolate=False, step=None, **kwargs):
        return self._fill_between_x_or_y('x', x, y1, y2, where=where, interpolate=interpolate, step=step, **kwargs)
    if _fill_between_x_or_y.__doc__:
        fill_between.__doc__ = _fill_between_x_or_y.__doc__.format(dir='horizontal', ind='x', dep='y')
    fill_between = _preprocess_data(_docstring.interpd(fill_between), replace_names=['x', 'y1', 'y2', 'where'])

    def fill_betweenx(self, y, x1, x2=0, where=None, step=None, interpolate=False, **kwargs):
        return self._fill_between_x_or_y('y', y, x1, x2, where=where, interpolate=interpolate, step=step, **kwargs)
    if _fill_between_x_or_y.__doc__:
        fill_betweenx.__doc__ = _fill_between_x_or_y.__doc__.format(dir='vertical', ind='y', dep='x')
    fill_betweenx = _preprocess_data(_docstring.interpd(fill_betweenx), replace_names=['y', 'x1', 'x2', 'where'])

    @_api.make_keyword_only('3.10', 'vert')
    def violin(self, vpstats, positions=None, vert=None, orientation='vertical', widths=0.5, showmeans=False, showextrema=True, showmedians=False, side='both', facecolor=None, linecolor=None):
        """
        Draw a violin plot from pre-computed statistics.

        Draw a violin plot for each column of *vpstats*. Each filled area
        extends to represent the entire data range, with optional lines at the
        mean, the median, the minimum, the maximum, and the quantiles values.

        Parameters
        ----------
        vpstats : list of dicts
            A list of dictionaries containing stats for each violin plot.
            Required keys are:

            - ``coords``: A list of scalars containing the coordinates that
              the violin's kernel density estimate were evaluated at.

            - ``vals``: A list of scalars containing the values of the
              kernel density estimate at each of the coordinates given
              in *coords*.

            - ``mean``: The mean value for this violin's dataset.

            - ``median``: The median value for this violin's dataset.

            - ``min``: The minimum value for this violin's dataset.

            - ``max``: The maximum value for this violin's dataset.

            Optional keys are:

            - ``quantiles``: A list of scalars containing the quantile values
              for this violin's dataset.

        positions : array-like, default: [1, 2, ..., n]
            The positions of the violins; i.e. coordinates on the x-axis for
            vertical violins (or y-axis for horizontal violins).

        vert : bool, optional
            .. deprecated:: 3.10
                Use *orientation* instead.

                If this is given during the deprecation period, it overrides
                the *orientation* parameter.

            If True, plots the violins vertically.
            If False, plots the violins horizontally.

        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            If 'horizontal', plots the violins horizontally.
            Otherwise, plots the violins vertically.

            .. versionadded:: 3.10

        widths : float or array-like, default: 0.5
            The maximum width of each violin in units of the *positions* axis.
            The default is 0.5, which is half available space when using default
            *positions*.

        showmeans : bool, default: False
            Whether to show the mean with a line.

        showextrema : bool, default: True
            Whether to show extrema with a line.

        showmedians : bool, default: False
            Whether to show the median with a line.

        side : {'both', 'low', 'high'}, default: 'both'
            'both' plots standard violins. 'low'/'high' only
            plots the side below/above the positions value.

        facecolor : :mpltype:`color` or list of :mpltype:`color`, optional
            If provided, will set the face color(s) of the violins.

            .. versionadded:: 3.11

            For backward compatibility, if *facecolor* is not given, the body
            will get an Artist-level transparency `alpha <.Artist.set_alpha>`
            of 0.3, which will persist if you afterwards change the facecolor,
            e.g. via ``result['bodies'][0].set_facecolor('red')``.
            If *facecolor* is given, there is no Artist-level transparency.
            To set transparency for *facecolor* or *edgecolor* use
            ``(color, alpha)`` tuples.

        linecolor : :mpltype:`color` or list of :mpltype:`color`, optional
            If provided, will set the line color(s) of the violins (the
            horizontal and vertical spines and body edges).

            .. versionadded:: 3.11

        Returns
        -------
        dict
            A dictionary mapping each component of the violinplot to a
            list of the corresponding collection instances created. The
            dictionary has the following keys:

            - ``bodies``: A list of the `~.collections.PolyCollection`
              instances containing the filled area of each violin.

            - ``cmeans``: A `~.collections.LineCollection` instance that marks
              the mean values of each of the violin's distribution.

            - ``cmins``: A `~.collections.LineCollection` instance that marks
              the bottom of each violin's distribution.

            - ``cmaxes``: A `~.collections.LineCollection` instance that marks
              the top of each violin's distribution.

            - ``cbars``: A `~.collections.LineCollection` instance that marks
              the centers of each violin's distribution.

            - ``cmedians``: A `~.collections.LineCollection` instance that
              marks the median values of each of the violin's distribution.

            - ``cquantiles``: A `~.collections.LineCollection` instance created
              to identify the quantiles values of each of the violin's
              distribution.

        See Also
        --------
        violinplot :
            Draw a violin plot from data instead of pre-computed statistics.
        .cbook.violin_stats:
            Calculate a *vpstats* dictionary from data, suitable for passing to violin.
        """
        means = []
        mins = []
        maxes = []
        medians = []
        quantiles = []
        qlens = []
        artists = {}
        N = len(vpstats)
        datashape_message = 'List of violinplot statistics and `{0}` values must have the same length'
        if vert is not None:
            _api.warn_deprecated('3.11', name='vert: bool', alternative="orientation: {'vertical', 'horizontal'}")
            orientation = 'vertical' if vert else 'horizontal'
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)
        if positions is None:
            positions = range(1, N + 1)
        elif len(positions) != N:
            raise ValueError(datashape_message.format('positions'))
        if np.isscalar(widths):
            widths = [widths] * N
        elif len(widths) != N:
            raise ValueError(datashape_message.format('widths'))
        _api.check_in_list(['both', 'low', 'high'], side=side)
        line_ends = [[-0.25 if side in ['both', 'low'] else 0], [0.25 if side in ['both', 'high'] else 0]] * np.array(widths) + positions

        def cycle_color(color, alpha=None):
            rgba = mcolors.to_rgba_array(color, alpha=alpha)
            color_cycler = itertools.chain(itertools.cycle(rgba), itertools.repeat('none'))
            color_list = []
            for _ in range(N):
                color_list.append(next(color_cycler))
            return color_list
        if facecolor is None or linecolor is None:
            if not mpl.rcParams['_internal.classic_mode']:
                next_color = self._get_lines.get_next_color()
        if facecolor is not None:
            facecolor = cycle_color(facecolor)
            body_artist_alpha = None
        else:
            body_artist_alpha = 0.3
            if mpl.rcParams['_internal.classic_mode']:
                facecolor = cycle_color('y')
            else:
                facecolor = cycle_color(next_color)
        if mpl.rcParams['_internal.classic_mode']:
            body_edgecolor = ('k', 0.3)
        else:
            body_edgecolor = None
        if linecolor is not None:
            linecolor = cycle_color(linecolor)
        elif mpl.rcParams['_internal.classic_mode']:
            linecolor = cycle_color('r')
        else:
            linecolor = cycle_color(next_color)
        if orientation == 'vertical':
            fill = self.fill_betweenx
            if side in ['low', 'high']:
                perp_lines = functools.partial(self.hlines, colors=linecolor, capstyle='projecting')
                par_lines = functools.partial(self.vlines, colors=linecolor, capstyle='projecting')
            else:
                perp_lines = functools.partial(self.hlines, colors=linecolor)
                par_lines = functools.partial(self.vlines, colors=linecolor)
        else:
            fill = self.fill_between
            if side in ['low', 'high']:
                perp_lines = functools.partial(self.vlines, colors=linecolor, capstyle='projecting')
                par_lines = functools.partial(self.hlines, colors=linecolor, capstyle='projecting')
            else:
                perp_lines = functools.partial(self.vlines, colors=linecolor)
                par_lines = functools.partial(self.hlines, colors=linecolor)
        bodies = []
        bodies_zip = zip(vpstats, positions, widths, facecolor)
        for stats, pos, width, facecolor in bodies_zip:
            vals = np.array(stats['vals'])
            vals = 0.5 * width * vals / vals.max()
            bodies += [fill(stats['coords'], -vals + pos if side in ['both', 'low'] else pos, vals + pos if side in ['both', 'high'] else pos, facecolor=facecolor, edgecolor=body_edgecolor, alpha=body_artist_alpha)]
            means.append(stats['mean'])
            mins.append(stats['min'])
            maxes.append(stats['max'])
            medians.append(stats['median'])
            q = stats.get('quantiles')
            if q is None:
                q = []
            quantiles.extend(q)
            qlens.append(len(q))
        artists['bodies'] = bodies
        if showmeans:
            artists['cmeans'] = perp_lines(means, *line_ends)
        if showextrema:
            artists['cmaxes'] = perp_lines(maxes, *line_ends)
            artists['cmins'] = perp_lines(mins, *line_ends)
            artists['cbars'] = par_lines(positions, mins, maxes)
        if showmedians:
            artists['cmedians'] = perp_lines(medians, *line_ends)
        if quantiles:
            artists['cquantiles'] = perp_lines(quantiles, *np.repeat(line_ends, qlens, axis=1))
        return artists
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
