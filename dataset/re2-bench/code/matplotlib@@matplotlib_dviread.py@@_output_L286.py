from functools import cache, cached_property, lru_cache, partial, wraps
import numpy as np

class Dvi:
    """
    A reader for a dvi ("device-independent") file, as produced by TeX.

    The current implementation can only iterate through pages in order,
    and does not even attempt to verify the postamble.

    This class can be used as a context manager to close the underlying
    file upon exit. Pages can be read via iteration. Here is an overly
    simple way to extract text without trying to detect whitespace::

        >>> with matplotlib.dviread.Dvi('input.dvi', 72) as dvi:
        ...     for page in dvi:
        ...         print(''.join(chr(t.glyph) for t in page.text))
    """
    _dtable = [None] * 256
    _dispatch = partial(_dispatch, _dtable)

    def __init__(self, filename, dpi):
        """
        Read the data from the file named *filename* and convert
        TeX's internal units to units of *dpi* per inch.
        *dpi* only sets the units and does not limit the resolution.
        Use None to return TeX's internal units.
        """
        _log.debug('Dvi: %s', filename)
        self.file = open(filename, 'rb')
        self.dpi = dpi
        self.fonts = {}
        self.state = _dvistate.pre
        self._missing_font = None

    def _output(self):
        """
        Output the text and boxes belonging to the most recent page.
        page = dvi._output()
        """
        minx = miny = np.inf
        maxx = maxy = -np.inf
        maxy_pure = -np.inf
        for elt in self.text + self.boxes:
            if isinstance(elt, Box):
                x, y, h, w = elt
                e = 0
            else:
                x, y, font, g, w = elt
                h, e = font._height_depth_of(g)
            minx = min(minx, x)
            miny = min(miny, y - h)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + e)
            maxy_pure = max(maxy_pure, y)
        if self._baseline_v is not None:
            maxy_pure = self._baseline_v
            self._baseline_v = None
        if not self.text and (not self.boxes):
            return Page(text=[], boxes=[], width=0, height=0, descent=0)
        if self.dpi is None:
            return Page(text=self.text, boxes=self.boxes, width=maxx - minx, height=maxy_pure - miny, descent=maxy - maxy_pure)
        d = self.dpi / (72.27 * 2 ** 16)
        descent = (maxy - maxy_pure) * d
        text = [Text((x - minx) * d, (maxy - y) * d - descent, f, g, w * d) for x, y, f, g, w in self.text]
        boxes = [Box((x - minx) * d, (maxy - y) * d - descent, h * d, w * d) for x, y, h, w in self.boxes]
        return Page(text=text, boxes=boxes, width=(maxx - minx) * d, height=(maxy_pure - miny) * d, descent=descent)
