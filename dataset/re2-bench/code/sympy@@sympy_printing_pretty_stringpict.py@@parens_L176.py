from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width, center

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

    def height(self):
        """The height of the picture in characters."""
        return len(self.picture)

    def parens(self, left='(', right=')', ifascii_nougly=False):
        """Put parentheses around self.
        Returns string, baseline arguments for stringPict.

        left or right can be None or empty string which means 'no paren from
        that side'
        """
        h = self.height()
        b = self.baseline
        if ifascii_nougly and (not pretty_use_unicode()):
            h = 1
            b = 0
        res = self
        if left:
            lparen = stringPict(vobj(left, h), baseline=b)
            res = stringPict(*lparen.right(self))
        if right:
            rparen = stringPict(vobj(right, h), baseline=b)
            res = stringPict(*res.right(rparen))
        return ('\n'.join(res.picture), res.baseline)
