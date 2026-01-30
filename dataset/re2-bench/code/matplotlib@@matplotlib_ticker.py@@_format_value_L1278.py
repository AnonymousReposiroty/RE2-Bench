import math
import numpy as np

class LogitFormatter(Formatter):
    """
    Probability formatter (using Math text).
    """

    def __init__(self, *, use_overline=False, one_half='\\frac{1}{2}', minor=False, minor_threshold=25, minor_number=6):
        """
        Parameters
        ----------
        use_overline : bool, default: False
            If x > 1/2, with x = 1 - v, indicate if x should be displayed as
            $\\overline{v}$. The default is to display $1 - v$.

        one_half : str, default: r"\\\\frac{1}{2}"
            The string used to represent 1/2.

        minor : bool, default: False
            Indicate if the formatter is formatting minor ticks or not.
            Basically minor ticks are not labelled, except when only few ticks
            are provided, ticks with most space with neighbor ticks are
            labelled. See other parameters to change the default behavior.

        minor_threshold : int, default: 25
            Maximum number of locs for labelling some minor ticks. This
            parameter have no effect if minor is False.

        minor_number : int, default: 6
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        """
        self._use_overline = use_overline
        self._one_half = one_half
        self._minor = minor
        self._labelled = set()
        self._minor_threshold = minor_threshold
        self._minor_number = minor_number

    def _format_value(self, x, locs, sci_notation=True):
        if sci_notation:
            exponent = math.floor(np.log10(x))
            min_precision = 0
        else:
            exponent = 0
            min_precision = 1
        value = x * 10 ** (-exponent)
        if len(locs) < 2:
            precision = min_precision
        else:
            diff = np.sort(np.abs(locs - x))[1]
            precision = -np.log10(diff) + exponent
            precision = int(np.round(precision)) if _is_close_to_int(precision) else math.ceil(precision)
            if precision < min_precision:
                precision = min_precision
        mantissa = '%.*f' % (precision, value)
        if not sci_notation:
            return mantissa
        s = '%s\\cdot10^{%d}' % (mantissa, exponent)
        return s
