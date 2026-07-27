import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
from matplotlib.rcsetup import cycler, validate_axisbelow

class _process_plot_var_args:
    """
    Process variable length arguments to `~.Axes.plot`, to support ::

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of *x*, *y*, *fmt* are allowed
    """

    def __init__(self, output='Line2D'):
        _api.check_in_list(['Line2D', 'Polygon', 'coordinates'], output=output)
        self.output = output
        self.set_prop_cycle(None)

    def set_prop_cycle(self, cycler):
        self._idx = 0
        self._cycler_items = [*mpl._val_or_rc(cycler, 'axes.prop_cycle')]

    def _getdefaults(self, kw, ignore=frozenset()):
        """
        If some keys in the property cycle (excluding those in the set
        *ignore*) are absent or set to None in the dict *kw*, return a copy
        of the next entry in the property cycle, excluding keys in *ignore*.
        Otherwise, don't advance the property cycle, and return an empty dict.
        """
        defaults = self._cycler_items[self._idx]
        if any((kw.get(k, None) is None for k in {*defaults} - ignore)):
            self._idx = (self._idx + 1) % len(self._cycler_items)
            return {k: v for k, v in defaults.items() if k not in ignore}
        else:
            return {}
