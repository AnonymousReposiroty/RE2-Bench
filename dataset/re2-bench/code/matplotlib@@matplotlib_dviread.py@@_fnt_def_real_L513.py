from functools import cache, cached_property, lru_cache, partial, wraps
from matplotlib import _api, cbook, font_manager

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

    def _fnt_def_real(self, k, c, s, d, a, l):
        n = self.file.read(a + l)
        fontname = n[-l:]
        if fontname.startswith(b'[') and c == 1282761030:
            self.fonts[k] = DviFont.from_luatex(s, n)
            return
        fontname = fontname.decode('ascii')
        try:
            tfm = _tfmfile(fontname)
        except FileNotFoundError as exc:
            if fontname.startswith('[') and fontname.endswith(';') and (c == 0):
                exc.add_note('This dvi file was likely generated with a too-old version of luaotfload; luaotfload 3.23 is required.')
            self.fonts[k] = cbook._ExceptionInfo.from_exception(exc)
            return
        if c != 0 and tfm.checksum != 0 and (c != tfm.checksum):
            raise ValueError(f'tfm checksum mismatch: {n}')
        try:
            vf = _vffile(fontname)
        except FileNotFoundError:
            vf = None
        self.fonts[k] = DviFont(scale=s, metrics=tfm, texname=n, vf=vf)
