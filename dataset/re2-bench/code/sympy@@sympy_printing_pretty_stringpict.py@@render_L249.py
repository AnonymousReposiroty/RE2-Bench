import shutil
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

    def width(self):
        """The width of the picture in characters."""
        return line_width(self.picture[0])

    def render(self, *args, **kwargs):
        """Return the string form of self.

           Unless the argument line_break is set to False, it will
           break the expression in a form that can be printed
           on the terminal without being broken up.
         """
        if _GLOBAL_WRAP_LINE is not None:
            kwargs['wrap_line'] = _GLOBAL_WRAP_LINE
        if kwargs['wrap_line'] is False:
            return '\n'.join(self.picture)
        if kwargs['num_columns'] is not None:
            ncols = kwargs['num_columns']
        else:
            ncols = self.terminal_width()
        if ncols <= 0:
            ncols = 80
        if self.width() <= ncols:
            return type(self.picture[0])(self)
        '\n        Break long-lines in a visually pleasing format.\n        without overflow indicators | with overflow indicators\n        |   2  2        3     |     |   2  2        3    ↪|\n        |6*x *y  + 4*x*y  +   |     |6*x *y  + 4*x*y  +  ↪|\n        |                     |     |                     |\n        |     3    4    4     |     |↪      3    4    4   |\n        |4*y*x  + x  + y      |     |↪ 4*y*x  + x  + y    |\n        |a*c*e + a*c*f + a*d  |     |a*c*e + a*c*f + a*d ↪|\n        |*e + a*d*f + b*c*e   |     |                     |\n        |+ b*c*f + b*d*e + b  |     |↪ *e + a*d*f + b*c* ↪|\n        |*d*f                 |     |                     |\n        |                     |     |↪ e + b*c*f + b*d*e ↪|\n        |                     |     |                     |\n        |                     |     |↪ + b*d*f            |\n        '
        overflow_first = ''
        if kwargs['use_unicode'] or pretty_use_unicode():
            overflow_start = '↪ '
            overflow_end = ' ↪'
        else:
            overflow_start = '> '
            overflow_end = ' >'

        def chunks(line):
            """Yields consecutive chunks of line_width ncols"""
            prefix = overflow_first
            width, start = (line_width(prefix + overflow_end), 0)
            for i, x in enumerate(line):
                wx = line_width(x)
                if width + wx > ncols:
                    yield (prefix + line[start:i] + overflow_end)
                    prefix = overflow_start
                    width, start = (line_width(prefix + overflow_end), i)
                width += wx
            yield (prefix + line[start:])
        pictures = zip(*map(chunks, self.picture))
        pictures = ['\n'.join(picture) for picture in pictures]
        return '\n\n'.join(pictures)

    def terminal_width(self):
        """Return the terminal width if possible, otherwise return 0.
        """
        size = shutil.get_terminal_size(fallback=(0, 0))
        return size.columns
