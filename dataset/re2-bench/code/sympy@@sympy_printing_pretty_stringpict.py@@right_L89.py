class stringPict:
    """An ASCII picture.
    The pictures are represented as a list of equal length strings.
    """
    LINE = 'line'

    def __init__(self, s, baseline=0):
        """Initialize from string.
        Multiline strings are centered.
        """
        self.s = s
        self.picture = stringPict.equalLengths(s.splitlines())
        self.baseline = baseline
        self.binding = None

    def right(self, *args):
        """Put pictures next to this one.
        Returns string, baseline arguments for stringPict.
        (Multiline) strings are allowed, and are given a baseline of 0.

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("10").right(" + ",stringPict("1\\r-\\r2",1))[0])
             1
        10 + -
             2

        """
        return stringPict.next(self, *args)
