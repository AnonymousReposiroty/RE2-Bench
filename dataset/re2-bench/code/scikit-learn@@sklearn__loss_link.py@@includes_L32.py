from dataclasses import dataclass
import numpy as np

@dataclass
class Interval:
    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool

    def includes(self, x):
        """Test whether all values of x are in interval range.

        Parameters
        ----------
        x : ndarray
            Array whose elements are tested to be in interval range.

        Returns
        -------
        result : bool
        """
        if self.low_inclusive:
            low = np.greater_equal(x, self.low)
        else:
            low = np.greater(x, self.low)
        if not np.all(low):
            return False
        if self.high_inclusive:
            high = np.less_equal(x, self.high)
        else:
            high = np.less(x, self.high)
        return bool(np.all(high))
