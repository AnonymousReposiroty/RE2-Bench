from functools import cache, cached_property, lru_cache, partial, wraps

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

    def close(self):
        """Close the underlying file if it is open."""
        if not self.file.closed:
            self.file.close()

    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        down_stack = [0]
        self._baseline_v = None
        while True:
            byte = self.file.read(1)[0]
            self._dtable[byte](self, byte)
            if self._missing_font:
                raise self._missing_font.to_exception()
            name = self._dtable[byte].__name__
            if name == '_push':
                down_stack.append(down_stack[-1])
            elif name == '_pop':
                down_stack.pop()
            elif name == '_down':
                down_stack[-1] += 1
            if self._baseline_v is None and len(getattr(self, 'stack', [])) == 3 and (down_stack[-1] >= 4):
                self._baseline_v = self.v
            if byte == 140:
                return True
            if self.state is _dvistate.post_post:
                self.close()
                return False
