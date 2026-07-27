import matplotlib as mpl
from matplotlib._constrained_layout import do_constrained_layout

class ConstrainedLayoutEngine(LayoutEngine):
    """
    Implements the ``constrained_layout`` geometry management.  See
    :ref:`constrainedlayout_guide` for details.
    """
    _adjust_compatible = False
    _colorbar_gridspec = False

    def __init__(self, *, h_pad=None, w_pad=None, hspace=None, wspace=None, rect=(0, 0, 1, 1), compress=False, **kwargs):
        """
        Initialize ``constrained_layout`` settings.

        Parameters
        ----------
        h_pad, w_pad : float
            Padding around the Axes elements in inches.
            Default to :rc:`figure.constrained_layout.h_pad` and
            :rc:`figure.constrained_layout.w_pad`.
        hspace, wspace : float
            Fraction of the figure to dedicate to space between the
            axes.  These are evenly spread between the gaps between the Axes.
            A value of 0.2 for a three-column layout would have a space
            of 0.1 of the figure width between each column.
            If h/wspace < h/w_pad, then the pads are used instead.
            Default to :rc:`figure.constrained_layout.hspace` and
            :rc:`figure.constrained_layout.wspace`.
        rect : tuple of 4 floats
            Rectangle in figure coordinates to perform constrained layout in
            (left, bottom, width, height), each from 0-1.
        compress : bool
            Whether to shift Axes so that white space in between them is
            removed. This is useful for simple grids of fixed-aspect Axes (e.g.
            a grid of images).  See :ref:`compressed_layout`.
        """
        super().__init__(**kwargs)
        self.set(w_pad=mpl.rcParams['figure.constrained_layout.w_pad'], h_pad=mpl.rcParams['figure.constrained_layout.h_pad'], wspace=mpl.rcParams['figure.constrained_layout.wspace'], hspace=mpl.rcParams['figure.constrained_layout.hspace'], rect=(0, 0, 1, 1))
        self.set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace, rect=rect)
        self._compress = compress

    def execute(self, fig):
        """
        Perform constrained_layout and move and resize Axes accordingly.

        Parameters
        ----------
        fig : `.Figure` to perform layout on.
        """
        width, height = fig.get_size_inches()
        w_pad = self._params['w_pad'] / width
        h_pad = self._params['h_pad'] / height
        return do_constrained_layout(fig, w_pad=w_pad, h_pad=h_pad, wspace=self._params['wspace'], hspace=self._params['hspace'], rect=self._params['rect'], compress=self._compress)

    def set(self, *, h_pad=None, w_pad=None, hspace=None, wspace=None, rect=None):
        """
        Set the pads for constrained_layout.

        Parameters
        ----------
        h_pad, w_pad : float
            Padding around the Axes elements in inches.
            Default to :rc:`figure.constrained_layout.h_pad` and
            :rc:`figure.constrained_layout.w_pad`.
        hspace, wspace : float
            Fraction of the figure to dedicate to space between the
            axes.  These are evenly spread between the gaps between the Axes.
            A value of 0.2 for a three-column layout would have a space
            of 0.1 of the figure width between each column.
            If h/wspace < h/w_pad, then the pads are used instead.
            Default to :rc:`figure.constrained_layout.hspace` and
            :rc:`figure.constrained_layout.wspace`.
        rect : tuple of 4 floats
            Rectangle in figure coordinates to perform constrained layout in
            (left, bottom, width, height), each from 0-1.
        """
        for td in self.set.__kwdefaults__:
            if locals()[td] is not None:
                self._params[td] = locals()[td]
