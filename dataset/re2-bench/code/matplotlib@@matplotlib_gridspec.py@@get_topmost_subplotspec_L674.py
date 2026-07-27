class SubplotSpec:
    """
    The location of a subplot in a `GridSpec`.

    .. note::

        Likely, you will never instantiate a `SubplotSpec` yourself. Instead,
        you will typically obtain one from a `GridSpec` using item-access.

    Parameters
    ----------
    gridspec : `~matplotlib.gridspec.GridSpec`
        The GridSpec, which the subplot is referencing.
    num1, num2 : int
        The subplot will occupy the *num1*-th cell of the given
        *gridspec*.  If *num2* is provided, the subplot will span between
        *num1*-th cell and *num2*-th cell **inclusive**.

        The index starts from 0.
    """

    def __init__(self, gridspec, num1, num2=None):
        self._gridspec = gridspec
        self.num1 = num1
        self.num2 = num2

    @property
    def num2(self):
        return self.num1 if self._num2 is None else self._num2

    @num2.setter
    def num2(self, value):
        self._num2 = value

    def get_gridspec(self):
        return self._gridspec

    def get_topmost_subplotspec(self):
        """
        Return the topmost `SubplotSpec` instance associated with the subplot.
        """
        gridspec = self.get_gridspec()
        if hasattr(gridspec, 'get_topmost_subplotspec'):
            return gridspec.get_topmost_subplotspec()
        else:
            return self
