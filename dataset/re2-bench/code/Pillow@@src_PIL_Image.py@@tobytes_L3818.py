import struct
import warnings
from . import (
    ExifTags,
    ImageMode,
    TiffTags,
    UnidentifiedImageError,
    __version__,
    _plugins,
)
from ._binary import i32le, o32be, o32le
from . import TiffImagePlugin
from . import TiffImagePlugin
from . import TiffImagePlugin
from . import TiffImagePlugin
from .TiffImagePlugin import ImageFileDirectory_v2

class Exif(_ExifBase):
    """
    This class provides read and write access to EXIF image data::

      from PIL import Image
      im = Image.open("exif.png")
      exif = im.getexif()  # Returns an instance of this class

    Information can be read and written, iterated over or deleted::

      print(exif[274])  # 1
      exif[274] = 2
      for k, v in exif.items():
        print("Tag", k, "Value", v)  # Tag 274 Value 2
      del exif[274]

    To access information beyond IFD0, :py:meth:`~PIL.Image.Exif.get_ifd`
    returns a dictionary::

      from PIL import ExifTags
      im = Image.open("exif_gps.jpg")
      exif = im.getexif()
      gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
      print(gps_ifd)

    Other IFDs include ``ExifTags.IFD.Exif``, ``ExifTags.IFD.Makernote``,
    ``ExifTags.IFD.Interop`` and ``ExifTags.IFD.IFD1``.

    :py:mod:`~PIL.ExifTags` also has enum classes to provide names for data::

      print(exif[ExifTags.Base.Software])  # PIL
      print(gps_ifd[ExifTags.GPS.GPSDateStamp])  # 1999:99:99 99:99:99
    """
    endian = None
    bigtiff = False
    _loaded = False

    def __init__(self):
        self._data = {}
        self._hidden_data = {}
        self._ifds = {}
        self._info = None
        self._loaded_exif = None

    def _fixup(self, value):
        try:
            if len(value) == 1 and isinstance(value, tuple):
                return value[0]
        except Exception:
            pass
        return value

    def _fixup_dict(self, src_dict):
        return {k: self._fixup(v) for k, v in src_dict.items()}

    def _get_ifd_dict(self, offset, group=None):
        try:
            self.fp.seek(offset)
        except (KeyError, TypeError):
            pass
        else:
            from . import TiffImagePlugin
            info = TiffImagePlugin.ImageFileDirectory_v2(self.head, group=group)
            info.load(self.fp)
            return self._fixup_dict(info)

    def _get_head(self):
        version = b'+' if self.bigtiff else b'*'
        if self.endian == '<':
            head = b'II' + version + b'\x00' + o32le(8)
        else:
            head = b'MM\x00' + version + o32be(8)
        if self.bigtiff:
            head += o32le(8) if self.endian == '<' else o32be(8)
            head += b'\x00\x00\x00\x00'
        return head

    def tobytes(self, offset: int=8) -> bytes:
        from . import TiffImagePlugin
        head = self._get_head()
        ifd = TiffImagePlugin.ImageFileDirectory_v2(ifh=head)
        for tag, value in self.items():
            if tag in [ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo] and (not isinstance(value, dict)):
                value = self.get_ifd(tag)
                if tag == ExifTags.IFD.Exif and ExifTags.IFD.Interop in value and (not isinstance(value[ExifTags.IFD.Interop], dict)):
                    value = value.copy()
                    value[ExifTags.IFD.Interop] = self.get_ifd(ExifTags.IFD.Interop)
            ifd[tag] = value
        return b'Exif\x00\x00' + head + ifd.tobytes(offset)

    def get_ifd(self, tag):
        if tag not in self._ifds:
            if tag == ExifTags.IFD.IFD1:
                if self._info is not None and self._info.next != 0:
                    self._ifds[tag] = self._get_ifd_dict(self._info.next)
            elif tag in [ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo]:
                offset = self._hidden_data.get(tag, self.get(tag))
                if offset is not None:
                    self._ifds[tag] = self._get_ifd_dict(offset, tag)
            elif tag in [ExifTags.IFD.Interop, ExifTags.IFD.Makernote]:
                if ExifTags.IFD.Exif not in self._ifds:
                    self.get_ifd(ExifTags.IFD.Exif)
                tag_data = self._ifds[ExifTags.IFD.Exif][tag]
                if tag == ExifTags.IFD.Makernote:
                    from .TiffImagePlugin import ImageFileDirectory_v2
                    if tag_data[:8] == b'FUJIFILM':
                        ifd_offset = i32le(tag_data, 8)
                        ifd_data = tag_data[ifd_offset:]
                        makernote = {}
                        for i in range(0, struct.unpack('<H', ifd_data[:2])[0]):
                            ifd_tag, typ, count, data = struct.unpack('<HHL4s', ifd_data[i * 12 + 2:(i + 1) * 12 + 2])
                            try:
                                unit_size, handler = ImageFileDirectory_v2._load_dispatch[typ]
                            except KeyError:
                                continue
                            size = count * unit_size
                            if size > 4:
                                offset, = struct.unpack('<L', data)
                                data = ifd_data[offset - 12:offset + size - 12]
                            else:
                                data = data[:size]
                            if len(data) != size:
                                warnings.warn(f'Possibly corrupt EXIF MakerNote data.  Expecting to read {size} bytes but only got {len(data)}. Skipping tag {ifd_tag}')
                                continue
                            if not data:
                                continue
                            makernote[ifd_tag] = handler(ImageFileDirectory_v2(), data, False)
                        self._ifds[tag] = dict(self._fixup_dict(makernote))
                    elif self.get(271) == 'Nintendo':
                        makernote = {}
                        for i in range(0, struct.unpack('>H', tag_data[:2])[0]):
                            ifd_tag, typ, count, data = struct.unpack('>HHL4s', tag_data[i * 12 + 2:(i + 1) * 12 + 2])
                            if ifd_tag == 4353:
                                offset, = struct.unpack('>L', data)
                                self.fp.seek(offset)
                                camerainfo = {'ModelID': self.fp.read(4)}
                                self.fp.read(4)
                                camerainfo['TimeStamp'] = i32le(self.fp.read(12))
                                self.fp.read(4)
                                camerainfo['InternalSerialNumber'] = self.fp.read(4)
                                self.fp.read(12)
                                parallax = self.fp.read(4)
                                handler = ImageFileDirectory_v2._load_dispatch[TiffTags.FLOAT][1]
                                camerainfo['Parallax'] = handler(ImageFileDirectory_v2(), parallax, False)
                                self.fp.read(4)
                                camerainfo['Category'] = self.fp.read(2)
                                makernote = {4353: dict(self._fixup_dict(camerainfo))}
                        self._ifds[tag] = makernote
                else:
                    self._ifds[tag] = self._get_ifd_dict(tag_data, tag)
        ifd = self._ifds.get(tag, {})
        if tag == ExifTags.IFD.Exif and self._hidden_data:
            ifd = {k: v for k, v in ifd.items() if k not in (ExifTags.IFD.Interop, ExifTags.IFD.Makernote)}
        return ifd
