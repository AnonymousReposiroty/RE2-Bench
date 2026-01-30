from numbers import Integral, Real

class Interval(_Constraint):
    """Constraint representing a typed interval.

    Parameters
    ----------
    type : {numbers.Integral, numbers.Real, RealNotInt}
        The set of numbers in which to set the interval.

        If RealNotInt, only reals that don't have the integer type
        are allowed. For example 1.0 is allowed but 1 is not.

    left : float or int or None
        The left bound of the interval. None means left bound is -∞.

    right : float, int or None
        The right bound of the interval. None means right bound is +∞.

    closed : {"left", "right", "both", "neither"}
        Whether the interval is open or closed. Possible choices are:

        - `"left"`: the interval is closed on the left and open on the right.
          It is equivalent to the interval `[ left, right )`.
        - `"right"`: the interval is closed on the right and open on the left.
          It is equivalent to the interval `( left, right ]`.
        - `"both"`: the interval is closed.
          It is equivalent to the interval `[ left, right ]`.
        - `"neither"`: the interval is open.
          It is equivalent to the interval `( left, right )`.

    Notes
    -----
    Setting a bound to `None` and setting the interval closed is valid. For instance,
    strictly speaking, `Interval(Real, 0, None, closed="both")` corresponds to
    `[0, +∞) U {+∞}`.
    """

    def __init__(self, type, left, right, *, closed):
        super().__init__()
        self.type = type
        self.left = left
        self.right = right
        self.closed = closed
        self._check_params()

    def _check_params(self):
        if self.type not in (Integral, Real, RealNotInt):
            raise ValueError(f'type must be either numbers.Integral, numbers.Real or RealNotInt. Got {self.type} instead.')
        if self.closed not in ('left', 'right', 'both', 'neither'):
            raise ValueError(f"closed must be either 'left', 'right', 'both' or 'neither'. Got {self.closed} instead.")
        if self.type is Integral:
            suffix = 'for an interval over the integers.'
            if self.left is not None and (not isinstance(self.left, Integral)):
                raise TypeError(f'Expecting left to be an int {suffix}')
            if self.right is not None and (not isinstance(self.right, Integral)):
                raise TypeError(f'Expecting right to be an int {suffix}')
            if self.left is None and self.closed in ('left', 'both'):
                raise ValueError(f"left can't be None when closed == {self.closed} {suffix}")
            if self.right is None and self.closed in ('right', 'both'):
                raise ValueError(f"right can't be None when closed == {self.closed} {suffix}")
        else:
            if self.left is not None and (not isinstance(self.left, Real)):
                raise TypeError('Expecting left to be a real number.')
            if self.right is not None and (not isinstance(self.right, Real)):
                raise TypeError('Expecting right to be a real number.')
        if self.right is not None and self.left is not None and (self.right <= self.left):
            raise ValueError(f"right can't be less than left. Got left={self.left} and right={self.right}")

    def __str__(self):
        type_str = 'an int' if self.type is Integral else 'a float'
        left_bracket = '[' if self.closed in ('left', 'both') else '('
        left_bound = '-inf' if self.left is None else self.left
        right_bound = 'inf' if self.right is None else self.right
        right_bracket = ']' if self.closed in ('right', 'both') else ')'
        if not self.type == Integral and isinstance(self.left, Real):
            left_bound = float(left_bound)
        if not self.type == Integral and isinstance(self.right, Real):
            right_bound = float(right_bound)
        return f'{type_str} in the range {left_bracket}{left_bound}, {right_bound}{right_bracket}'
