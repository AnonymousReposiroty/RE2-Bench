from numbers import Integral
import numpy as np

class GridSpecBase:
    """
    A base class of GridSpec that specifies the geometry of the grid
    that a subplot will be placed.
    """

    def __init__(self, nrows, ncols, height_ratios=None, width_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.
        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.
        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
        if not isinstance(nrows, Integral) or nrows <= 0:
            raise ValueError(f'Number of rows must be a positive integer, not {nrows!r}')
        if not isinstance(ncols, Integral) or ncols <= 0:
            raise ValueError(f'Number of columns must be a positive integer, not {ncols!r}')
        self._nrows, self._ncols = (nrows, ncols)
        self.set_height_ratios(height_ratios)
        self.set_width_ratios(width_ratios)
    nrows = property(lambda self: self._nrows, doc='The number of rows in the grid.')
    ncols = property(lambda self: self._ncols, doc='The number of columns in the grid.')

    def get_geometry(self):
        """
        Return a tuple containing the number of rows and columns in the grid.
        """
        return (self._nrows, self._ncols)

    def get_subplot_params(self, figure=None):
        pass

    def set_width_ratios(self, width_ratios):
        """
        Set the relative widths of the columns.

        *width_ratios* must be of length *ncols*. Each column gets a relative
        width of ``width_ratios[i] / sum(width_ratios)``.
        """
        if width_ratios is None:
            width_ratios = [1] * self._ncols
        elif len(width_ratios) != self._ncols:
            raise ValueError('Expected the given number of width ratios to match the number of columns of the grid')
        self._col_width_ratios = width_ratios

    def set_height_ratios(self, height_ratios):
        """
        Set the relative heights of the rows.

        *height_ratios* must be of length *nrows*. Each row gets a relative
        height of ``height_ratios[i] / sum(height_ratios)``.
        """
        if height_ratios is None:
            height_ratios = [1] * self._nrows
        elif len(height_ratios) != self._nrows:
            raise ValueError('Expected the given number of height ratios to match the number of rows of the grid')
        self._row_height_ratios = height_ratios

    def get_grid_positions(self, fig):
        """
        Return the positions of the grid cells in figure coordinates.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure the grid should be applied to. The subplot parameters
            (margins and spacing between subplots) are taken from *fig*.

        Returns
        -------
        bottoms, tops, lefts, rights : array
            The bottom, top, left, right positions of the grid cells in
            figure coordinates.
        """
        nrows, ncols = self.get_geometry()
        subplot_params = self.get_subplot_params(fig)
        left = subplot_params.left
        right = subplot_params.right
        bottom = subplot_params.bottom
        top = subplot_params.top
        wspace = subplot_params.wspace
        hspace = subplot_params.hspace
        tot_width = right - left
        tot_height = top - bottom
        cell_h = tot_height / (nrows + hspace * (nrows - 1))
        sep_h = hspace * cell_h
        norm = cell_h * nrows / sum(self._row_height_ratios)
        cell_heights = [r * norm for r in self._row_height_ratios]
        sep_heights = [0] + [sep_h] * (nrows - 1)
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)
        cell_w = tot_width / (ncols + wspace * (ncols - 1))
        sep_w = wspace * cell_w
        norm = cell_w * ncols / sum(self._col_width_ratios)
        cell_widths = [r * norm for r in self._col_width_ratios]
        sep_widths = [0] + [sep_w] * (ncols - 1)
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)
        fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T
        fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T
        return (fig_bottoms, fig_tops, fig_lefts, fig_rights)
