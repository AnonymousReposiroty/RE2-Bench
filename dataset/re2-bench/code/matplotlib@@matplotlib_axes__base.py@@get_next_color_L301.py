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

    def get_next_color(self):
        """Return the next color in the cycle."""
        entry = self._cycler_items[self._idx]
        if 'color' in entry:
            self._idx = (self._idx + 1) % len(self._cycler_items)
            return entry['color']
        else:
            return 'k'
