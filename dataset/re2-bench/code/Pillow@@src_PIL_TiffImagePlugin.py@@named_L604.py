import struct
from typing import TYPE_CHECKING, Any, Callable
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags

class ImageFileDirectory_v2(_IFDv2Base):
    """This class represents a TIFF tag directory.  To speed things up, we
    don't decode tags unless they're asked for.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v2()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        'Some Data'

    Individual values are returned as the strings or numbers, sequences are
    returned as tuples of the values.

    The tiff metadata type of each item is stored in a dictionary of
    tag types in
    :attr:`~PIL.TiffImagePlugin.ImageFileDirectory_v2.tagtype`. The types
    are read from a tiff file, guessed from the type added, or added
    manually.

    Data Structures:

        * ``self.tagtype = {}``

          * Key: numerical TIFF tag number
          * Value: integer corresponding to the data type from
            :py:data:`.TiffTags.TYPES`

          .. versionadded:: 3.0.0

    'Internal' data structures:

        * ``self._tags_v2 = {}``

          * Key: numerical TIFF tag number
          * Value: decoded data, as tuple for multiple values

        * ``self._tagdata = {}``

          * Key: numerical TIFF tag number
          * Value: undecoded byte string from file

        * ``self._tags_v1 = {}``

          * Key: numerical TIFF tag number
          * Value: decoded data in the v1 format

    Tags will be found in the private attributes ``self._tagdata``, and in
    ``self._tags_v2`` once decoded.

    ``self.legacy_api`` is a value for internal use, and shouldn't be changed
    from outside code. In cooperation with
    :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`, if ``legacy_api``
    is true, then decoded tags will be populated into both ``_tags_v1`` and
    ``_tags_v2``. ``_tags_v2`` will be used if this IFD is used in the TIFF
    save routine. Tags should be read from ``_tags_v1`` if
    ``legacy_api == true``.

    """
    _load_dispatch: dict[int, Callable[[ImageFileDirectory_v2, bytes, bool], Any]] = {}
    _write_dispatch: dict[int, Callable[..., Any]] = {}

    def __init__(self, ifh=b'II*\x00\x00\x00\x00\x00', prefix=None, group=None):
        """Initialize an ImageFileDirectory.

        To construct an ImageFileDirectory from a real file, pass the 8-byte
        magic header to the constructor.  To only set the endianness, pass it
        as the 'prefix' keyword argument.

        :param ifh: One of the accepted magic headers (cf. PREFIXES); also sets
              endianness.
        :param prefix: Override the endianness of the file.
        """
        if not _accept(ifh):
            msg = f'not a TIFF file (header {repr(ifh)} not valid)'
            raise SyntaxError(msg)
        self._prefix = prefix if prefix is not None else ifh[:2]
        if self._prefix == MM:
            self._endian = '>'
        elif self._prefix == II:
            self._endian = '<'
        else:
            msg = 'not a TIFF IFD'
            raise SyntaxError(msg)
        self._bigtiff = ifh[2] == 43
        self.group = group
        self.tagtype = {}
        ' Dictionary of tag types '
        self.reset()
        self.next, = self._unpack('Q', ifh[8:]) if self._bigtiff else self._unpack('L', ifh[4:])
        self._legacy_api = False
    prefix = property(lambda self: self._prefix)
    offset = property(lambda self: self._offset)

    def reset(self):
        self._tags_v1 = {}
        self._tags_v2 = {}
        self._tagdata = {}
        self.tagtype = {}
        self._next = None
        self._offset = None

    def named(self):
        """
        :returns: dict of name|key: value

        Returns the complete tag dictionary, with named tags where possible.
        """
        return {TiffTags.lookup(code, self.group).name: value for code, value in self.items()}

    def _unpack(self, fmt, data):
        return struct.unpack(self._endian + fmt, data)
    list(map(_register_basic, [(TiffTags.SHORT, 'H', 'short'), (TiffTags.LONG, 'L', 'long'), (TiffTags.SIGNED_BYTE, 'b', 'signed byte'), (TiffTags.SIGNED_SHORT, 'h', 'signed short'), (TiffTags.SIGNED_LONG, 'l', 'signed long'), (TiffTags.FLOAT, 'f', 'float'), (TiffTags.DOUBLE, 'd', 'double'), (TiffTags.IFD, 'L', 'long'), (TiffTags.LONG8, 'Q', 'long8')]))
