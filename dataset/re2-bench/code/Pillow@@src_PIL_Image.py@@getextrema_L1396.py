import cffi
from . import PyAccess

class Image:
    """
    This class represents an image object.  To create
    :py:class:`~PIL.Image.Image` objects, use the appropriate factory
    functions.  There's hardly ever any reason to call the Image constructor
    directly.

    * :py:func:`~PIL.Image.open`
    * :py:func:`~PIL.Image.new`
    * :py:func:`~PIL.Image.frombytes`
    """
    format: str | None = None
    format_description: str | None = None
    _close_exclusive_fp_after_loading = True

    def __init__(self):
        self.im = None
        self._mode = ''
        self._size = (0, 0)
        self.palette = None
        self.info = {}
        self.readonly = 0
        self.pyaccess = None
        self._exif = None

    def load(self):
        """
        Allocates storage for the image and loads the pixel data.  In
        normal cases, you don't need to call this method, since the
        Image class automatically loads an opened image when it is
        accessed for the first time.

        If the file associated with the image was opened by Pillow, then this
        method will close it. The exception to this is if the image has
        multiple frames, in which case the file will be left open for seek
        operations. See :ref:`file-handling` for more information.

        :returns: An image access object.
        :rtype: :ref:`PixelAccess` or :py:class:`PIL.PyAccess`
        """
        if self.im is not None and self.palette and self.palette.dirty:
            mode, arr = self.palette.getdata()
            self.im.putpalette(mode, arr)
            self.palette.dirty = 0
            self.palette.rawmode = None
            if 'transparency' in self.info and mode in ('LA', 'PA'):
                if isinstance(self.info['transparency'], int):
                    self.im.putpalettealpha(self.info['transparency'], 0)
                else:
                    self.im.putpalettealphas(self.info['transparency'])
                self.palette.mode = 'RGBA'
            else:
                palette_mode = 'RGBA' if mode.startswith('RGBA') else 'RGB'
                self.palette.mode = palette_mode
                self.palette.palette = self.im.getpalette(palette_mode, palette_mode)
        if self.im is not None:
            if cffi and USE_CFFI_ACCESS:
                if self.pyaccess:
                    return self.pyaccess
                from . import PyAccess
                self.pyaccess = PyAccess.new(self, self.readonly)
                if self.pyaccess:
                    return self.pyaccess
            return self.im.pixel_access(self.readonly)
    __copy__ = copy

    def getextrema(self) -> tuple[float, float] | tuple[tuple[int, int], ...]:
        """
        Gets the minimum and maximum pixel values for each band in
        the image.

        :returns: For a single-band image, a 2-tuple containing the
           minimum and maximum pixel value.  For a multi-band image,
           a tuple containing one 2-tuple for each band.
        """
        self.load()
        if self.im.bands > 1:
            return tuple((self.im.getband(i).getextrema() for i in range(self.im.bands)))
        return self.im.getextrema()
