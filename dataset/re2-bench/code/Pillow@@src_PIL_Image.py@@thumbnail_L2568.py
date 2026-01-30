import math
import warnings
import cffi
from . import ImagePalette
from . import ImagePalette
from . import ImagePalette
from . import ImagePalette
from . import ImagePalette
from . import ImagePalette
from . import ImagePalette
from . import ImagePalette
from . import PyAccess
from . import ImageCms
from . import ImagePalette

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

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def mode(self):
        return self._mode

    def _new(self, im) -> Image:
        new = Image()
        new.im = im
        new._mode = im.mode
        new._size = im.size
        if im.mode in ('P', 'PA'):
            if self.palette:
                new.palette = self.palette.copy()
            else:
                from . import ImagePalette
                new.palette = ImagePalette.ImagePalette()
        new.info = self.info.copy()
        return new

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

    def convert(self, mode: str | None=None, matrix: tuple[float, ...] | None=None, dither: Dither | None=None, palette: Palette=Palette.WEB, colors: int=256) -> Image:
        """
        Returns a converted copy of this image. For the "P" mode, this
        method translates pixels through the palette.  If mode is
        omitted, a mode is chosen so that all information in the image
        and the palette can be represented without a palette.

        This supports all possible conversions between "L", "RGB" and "CMYK". The
        ``matrix`` argument only supports "L" and "RGB".

        When translating a color image to grayscale (mode "L"),
        the library uses the ITU-R 601-2 luma transform::

            L = R * 299/1000 + G * 587/1000 + B * 114/1000

        The default method of converting a grayscale ("L") or "RGB"
        image into a bilevel (mode "1") image uses Floyd-Steinberg
        dither to approximate the original image luminosity levels. If
        dither is ``None``, all values larger than 127 are set to 255 (white),
        all other values to 0 (black). To use other thresholds, use the
        :py:meth:`~PIL.Image.Image.point` method.

        When converting from "RGBA" to "P" without a ``matrix`` argument,
        this passes the operation to :py:meth:`~PIL.Image.Image.quantize`,
        and ``dither`` and ``palette`` are ignored.

        When converting from "PA", if an "RGBA" palette is present, the alpha
        channel from the image will be used instead of the values from the palette.

        :param mode: The requested mode. See: :ref:`concept-modes`.
        :param matrix: An optional conversion matrix.  If given, this
           should be 4- or 12-tuple containing floating point values.
        :param dither: Dithering method, used when converting from
           mode "RGB" to "P" or from "RGB" or "L" to "1".
           Available methods are :data:`Dither.NONE` or :data:`Dither.FLOYDSTEINBERG`
           (default). Note that this is not used when ``matrix`` is supplied.
        :param palette: Palette to use when converting from mode "RGB"
           to "P".  Available palettes are :data:`Palette.WEB` or
           :data:`Palette.ADAPTIVE`.
        :param colors: Number of colors to use for the :data:`Palette.ADAPTIVE`
           palette. Defaults to 256.
        :rtype: :py:class:`~PIL.Image.Image`
        :returns: An :py:class:`~PIL.Image.Image` object.
        """
        self.load()
        has_transparency = 'transparency' in self.info
        if not mode and self.mode == 'P':
            if self.palette:
                mode = self.palette.mode
            else:
                mode = 'RGB'
            if mode == 'RGB' and has_transparency:
                mode = 'RGBA'
        if not mode or (mode == self.mode and (not matrix)):
            return self.copy()
        if matrix:
            if mode not in ('L', 'RGB'):
                msg = 'illegal conversion'
                raise ValueError(msg)
            im = self.im.convert_matrix(mode, matrix)
            new_im = self._new(im)
            if has_transparency and self.im.bands == 3:
                transparency = new_im.info['transparency']

                def convert_transparency(m, v):
                    v = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * 0.5
                    return max(0, min(255, int(v)))
                if mode == 'L':
                    transparency = convert_transparency(matrix, transparency)
                elif len(mode) == 3:
                    transparency = tuple((convert_transparency(matrix[i * 4:i * 4 + 4], transparency) for i in range(0, len(transparency))))
                new_im.info['transparency'] = transparency
            return new_im
        if mode == 'P' and self.mode == 'RGBA':
            return self.quantize(colors)
        trns = None
        delete_trns = False
        if has_transparency:
            if self.mode in ('1', 'L', 'I', 'I;16') and mode in ('LA', 'RGBA') or (self.mode == 'RGB' and mode in ('La', 'LA', 'RGBa', 'RGBA')):
                new_im = self._new(self.im.convert_transparent(mode, self.info['transparency']))
                del new_im.info['transparency']
                return new_im
            elif self.mode in ('L', 'RGB', 'P') and mode in ('L', 'RGB', 'P'):
                t = self.info['transparency']
                if isinstance(t, bytes):
                    warnings.warn('Palette images with Transparency expressed in bytes should be converted to RGBA images')
                    delete_trns = True
                else:
                    trns_im = new(self.mode, (1, 1))
                    if self.mode == 'P':
                        trns_im.putpalette(self.palette)
                        if isinstance(t, tuple):
                            err = "Couldn't allocate a palette color for transparency"
                            try:
                                t = trns_im.palette.getcolor(t, self)
                            except ValueError as e:
                                if str(e) == 'cannot allocate more than 256 colors':
                                    t = None
                                else:
                                    raise ValueError(err) from e
                    if t is None:
                        trns = None
                    else:
                        trns_im.putpixel((0, 0), t)
                        if mode in ('L', 'RGB'):
                            trns_im = trns_im.convert(mode)
                        else:
                            trns_im = trns_im.convert('RGB')
                        trns = trns_im.getpixel((0, 0))
            elif self.mode == 'P' and mode in ('LA', 'PA', 'RGBA'):
                t = self.info['transparency']
                delete_trns = True
                if isinstance(t, bytes):
                    self.im.putpalettealphas(t)
                elif isinstance(t, int):
                    self.im.putpalettealpha(t, 0)
                else:
                    msg = 'Transparency for P mode should be bytes or int'
                    raise ValueError(msg)
        if mode == 'P' and palette == Palette.ADAPTIVE:
            im = self.im.quantize(colors)
            new_im = self._new(im)
            from . import ImagePalette
            new_im.palette = ImagePalette.ImagePalette('RGB', new_im.im.getpalette('RGB'))
            if delete_trns:
                del new_im.info['transparency']
            if trns is not None:
                try:
                    new_im.info['transparency'] = new_im.palette.getcolor(trns, new_im)
                except Exception:
                    del new_im.info['transparency']
                    warnings.warn("Couldn't allocate palette entry for transparency")
            return new_im
        if 'LAB' in (self.mode, mode):
            other_mode = mode if self.mode == 'LAB' else self.mode
            if other_mode in ('RGB', 'RGBA', 'RGBX'):
                from . import ImageCms
                srgb = ImageCms.createProfile('sRGB')
                lab = ImageCms.createProfile('LAB')
                profiles = [lab, srgb] if self.mode == 'LAB' else [srgb, lab]
                transform = ImageCms.buildTransform(profiles[0], profiles[1], self.mode, mode)
                return transform.apply(self)
        if dither is None:
            dither = Dither.FLOYDSTEINBERG
        try:
            im = self.im.convert(mode, dither)
        except ValueError:
            try:
                modebase = getmodebase(self.mode)
                if modebase == self.mode:
                    raise
                im = self.im.convert(modebase)
                im = im.convert(mode, dither)
            except KeyError as e:
                msg = 'illegal conversion'
                raise ValueError(msg) from e
        new_im = self._new(im)
        if mode == 'P' and palette != Palette.ADAPTIVE:
            from . import ImagePalette
            new_im.palette = ImagePalette.ImagePalette('RGB', im.getpalette('RGB'))
        if delete_trns:
            del new_im.info['transparency']
        if trns is not None:
            if new_im.mode == 'P':
                try:
                    new_im.info['transparency'] = new_im.palette.getcolor(trns, new_im)
                except ValueError as e:
                    del new_im.info['transparency']
                    if str(e) != 'cannot allocate more than 256 colors':
                        warnings.warn("Couldn't allocate palette entry for transparency")
            else:
                new_im.info['transparency'] = trns
        return new_im

    def quantize(self, colors: int=256, method: Quantize | None=None, kmeans: int=0, palette=None, dither: Dither=Dither.FLOYDSTEINBERG) -> Image:
        """
        Convert the image to 'P' mode with the specified number
        of colors.

        :param colors: The desired number of colors, <= 256
        :param method: :data:`Quantize.MEDIANCUT` (median cut),
                       :data:`Quantize.MAXCOVERAGE` (maximum coverage),
                       :data:`Quantize.FASTOCTREE` (fast octree),
                       :data:`Quantize.LIBIMAGEQUANT` (libimagequant; check support
                       using :py:func:`PIL.features.check_feature` with
                       ``feature="libimagequant"``).

                       By default, :data:`Quantize.MEDIANCUT` will be used.

                       The exception to this is RGBA images. :data:`Quantize.MEDIANCUT`
                       and :data:`Quantize.MAXCOVERAGE` do not support RGBA images, so
                       :data:`Quantize.FASTOCTREE` is used by default instead.
        :param kmeans: Integer greater than or equal to zero.
        :param palette: Quantize to the palette of given
                        :py:class:`PIL.Image.Image`.
        :param dither: Dithering method, used when converting from
           mode "RGB" to "P" or from "RGB" or "L" to "1".
           Available methods are :data:`Dither.NONE` or :data:`Dither.FLOYDSTEINBERG`
           (default).
        :returns: A new image
        """
        self.load()
        if method is None:
            method = Quantize.MEDIANCUT
            if self.mode == 'RGBA':
                method = Quantize.FASTOCTREE
        if self.mode == 'RGBA' and method not in (Quantize.FASTOCTREE, Quantize.LIBIMAGEQUANT):
            msg = 'Fast Octree (method == 2) and libimagequant (method == 3) are the only valid methods for quantizing RGBA images'
            raise ValueError(msg)
        if palette:
            palette.load()
            if palette.mode != 'P':
                msg = 'bad mode for palette image'
                raise ValueError(msg)
            if self.mode not in {'RGB', 'L'}:
                msg = 'only RGB or L mode images can be quantized to a palette'
                raise ValueError(msg)
            im = self.im.convert('P', dither, palette.im)
            new_im = self._new(im)
            new_im.palette = palette.palette.copy()
            return new_im
        if kmeans < 0:
            msg = 'kmeans must not be negative'
            raise ValueError(msg)
        im = self._new(self.im.quantize(colors, method, kmeans))
        from . import ImagePalette
        mode = im.im.getpalettemode()
        palette = im.im.getpalette(mode, mode)[:colors * len(mode)]
        im.palette = ImagePalette.ImagePalette(mode, palette)
        return im

    def copy(self) -> Image:
        """
        Copies this image. Use this method if you wish to paste things
        into an image, but still retain the original.

        :rtype: :py:class:`~PIL.Image.Image`
        :returns: An :py:class:`~PIL.Image.Image` object.
        """
        self.load()
        return self._new(self.im.copy())
    __copy__ = copy

    def draft(self, mode, size):
        """
        Configures the image file loader so it returns a version of the
        image that as closely as possible matches the given mode and
        size. For example, you can use this method to convert a color
        JPEG to grayscale while loading it.

        If any changes are made, returns a tuple with the chosen ``mode`` and
        ``box`` with coordinates of the original image within the altered one.

        Note that this method modifies the :py:class:`~PIL.Image.Image` object
        in place. If the image has already been loaded, this method has no
        effect.

        Note: This method is not implemented for most images. It is
        currently implemented only for JPEG and MPO images.

        :param mode: The requested mode.
        :param size: The requested size in pixels, as a 2-tuple:
           (width, height).
        """
        pass

    def _get_safe_box(self, size, resample, box):
        """Expands the box so it includes adjacent pixels
        that may be used by resampling with the given resampling filter.
        """
        filter_support = _filters_support[resample] - 0.5
        scale_x = (box[2] - box[0]) / size[0]
        scale_y = (box[3] - box[1]) / size[1]
        support_x = filter_support * scale_x
        support_y = filter_support * scale_y
        return (max(0, int(box[0] - support_x)), max(0, int(box[1] - support_y)), min(self.size[0], math.ceil(box[2] + support_x)), min(self.size[1], math.ceil(box[3] + support_y)))

    def resize(self, size, resample=None, box=None, reducing_gap=None) -> Image:
        """
        Returns a resized copy of this image.

        :param size: The requested size in pixels, as a 2-tuple:
           (width, height).
        :param resample: An optional resampling filter.  This can be
           one of :py:data:`Resampling.NEAREST`, :py:data:`Resampling.BOX`,
           :py:data:`Resampling.BILINEAR`, :py:data:`Resampling.HAMMING`,
           :py:data:`Resampling.BICUBIC` or :py:data:`Resampling.LANCZOS`.
           If the image has mode "1" or "P", it is always set to
           :py:data:`Resampling.NEAREST`. If the image mode specifies a number
           of bits, such as "I;16", then the default filter is
           :py:data:`Resampling.NEAREST`. Otherwise, the default filter is
           :py:data:`Resampling.BICUBIC`. See: :ref:`concept-filters`.
        :param box: An optional 4-tuple of floats providing
           the source image region to be scaled.
           The values must be within (0, 0, width, height) rectangle.
           If omitted or None, the entire source is used.
        :param reducing_gap: Apply optimization by resizing the image
           in two steps. First, reducing the image by integer times
           using :py:meth:`~PIL.Image.Image.reduce`.
           Second, resizing using regular resampling. The last step
           changes size no less than by ``reducing_gap`` times.
           ``reducing_gap`` may be None (no first step is performed)
           or should be greater than 1.0. The bigger ``reducing_gap``,
           the closer the result to the fair resampling.
           The smaller ``reducing_gap``, the faster resizing.
           With ``reducing_gap`` greater or equal to 3.0, the result is
           indistinguishable from fair resampling in most cases.
           The default value is None (no optimization).
        :returns: An :py:class:`~PIL.Image.Image` object.
        """
        if resample is None:
            type_special = ';' in self.mode
            resample = Resampling.NEAREST if type_special else Resampling.BICUBIC
        elif resample not in (Resampling.NEAREST, Resampling.BILINEAR, Resampling.BICUBIC, Resampling.LANCZOS, Resampling.BOX, Resampling.HAMMING):
            msg = f'Unknown resampling filter ({resample}).'
            filters = [f'{filter[1]} ({filter[0]})' for filter in ((Resampling.NEAREST, 'Image.Resampling.NEAREST'), (Resampling.LANCZOS, 'Image.Resampling.LANCZOS'), (Resampling.BILINEAR, 'Image.Resampling.BILINEAR'), (Resampling.BICUBIC, 'Image.Resampling.BICUBIC'), (Resampling.BOX, 'Image.Resampling.BOX'), (Resampling.HAMMING, 'Image.Resampling.HAMMING'))]
            msg += ' Use ' + ', '.join(filters[:-1]) + ' or ' + filters[-1]
            raise ValueError(msg)
        if reducing_gap is not None and reducing_gap < 1.0:
            msg = 'reducing_gap must be 1.0 or greater'
            raise ValueError(msg)
        size = tuple(size)
        self.load()
        if box is None:
            box = (0, 0) + self.size
        else:
            box = tuple(box)
        if self.size == size and box == (0, 0) + self.size:
            return self.copy()
        if self.mode in ('1', 'P'):
            resample = Resampling.NEAREST
        if self.mode in ['LA', 'RGBA'] and resample != Resampling.NEAREST:
            im = self.convert({'LA': 'La', 'RGBA': 'RGBa'}[self.mode])
            im = im.resize(size, resample, box)
            return im.convert(self.mode)
        self.load()
        if reducing_gap is not None and resample != Resampling.NEAREST:
            factor_x = int((box[2] - box[0]) / size[0] / reducing_gap) or 1
            factor_y = int((box[3] - box[1]) / size[1] / reducing_gap) or 1
            if factor_x > 1 or factor_y > 1:
                reduce_box = self._get_safe_box(size, resample, box)
                factor = (factor_x, factor_y)
                self = self.reduce(factor, box=reduce_box) if callable(self.reduce) else Image.reduce(self, factor, box=reduce_box)
                box = ((box[0] - reduce_box[0]) / factor_x, (box[1] - reduce_box[1]) / factor_y, (box[2] - reduce_box[0]) / factor_x, (box[3] - reduce_box[1]) / factor_y)
        return self._new(self.im.resize(size, resample, box))

    def reduce(self, factor, box=None):
        """
        Returns a copy of the image reduced ``factor`` times.
        If the size of the image is not dividable by ``factor``,
        the resulting size will be rounded up.

        :param factor: A greater than 0 integer or tuple of two integers
           for width and height separately.
        :param box: An optional 4-tuple of ints providing
           the source image region to be reduced.
           The values must be within ``(0, 0, width, height)`` rectangle.
           If omitted or ``None``, the entire source is used.
        """
        if not isinstance(factor, (list, tuple)):
            factor = (factor, factor)
        if box is None:
            box = (0, 0) + self.size
        else:
            box = tuple(box)
        if factor == (1, 1) and box == (0, 0) + self.size:
            return self.copy()
        if self.mode in ['LA', 'RGBA']:
            im = self.convert({'LA': 'La', 'RGBA': 'RGBa'}[self.mode])
            im = im.reduce(factor, box)
            return im.convert(self.mode)
        self.load()
        return self._new(self.im.reduce(factor, box))

    def thumbnail(self, size, resample=Resampling.BICUBIC, reducing_gap=2.0):
        """
        Make this image into a thumbnail.  This method modifies the
        image to contain a thumbnail version of itself, no larger than
        the given size.  This method calculates an appropriate thumbnail
        size to preserve the aspect of the image, calls the
        :py:meth:`~PIL.Image.Image.draft` method to configure the file reader
        (where applicable), and finally resizes the image.

        Note that this function modifies the :py:class:`~PIL.Image.Image`
        object in place.  If you need to use the full resolution image as well,
        apply this method to a :py:meth:`~PIL.Image.Image.copy` of the original
        image.

        :param size: The requested size in pixels, as a 2-tuple:
           (width, height).
        :param resample: Optional resampling filter.  This can be one
           of :py:data:`Resampling.NEAREST`, :py:data:`Resampling.BOX`,
           :py:data:`Resampling.BILINEAR`, :py:data:`Resampling.HAMMING`,
           :py:data:`Resampling.BICUBIC` or :py:data:`Resampling.LANCZOS`.
           If omitted, it defaults to :py:data:`Resampling.BICUBIC`.
           (was :py:data:`Resampling.NEAREST` prior to version 2.5.0).
           See: :ref:`concept-filters`.
        :param reducing_gap: Apply optimization by resizing the image
           in two steps. First, reducing the image by integer times
           using :py:meth:`~PIL.Image.Image.reduce` or
           :py:meth:`~PIL.Image.Image.draft` for JPEG images.
           Second, resizing using regular resampling. The last step
           changes size no less than by ``reducing_gap`` times.
           ``reducing_gap`` may be None (no first step is performed)
           or should be greater than 1.0. The bigger ``reducing_gap``,
           the closer the result to the fair resampling.
           The smaller ``reducing_gap``, the faster resizing.
           With ``reducing_gap`` greater or equal to 3.0, the result is
           indistinguishable from fair resampling in most cases.
           The default value is 2.0 (very close to fair resampling
           while still being faster in many cases).
        :returns: None
        """
        provided_size = tuple(map(math.floor, size))

        def preserve_aspect_ratio() -> tuple[int, int] | None:

            def round_aspect(number, key):
                return max(min(math.floor(number), math.ceil(number), key=key), 1)
            x, y = provided_size
            if x >= self.width and y >= self.height:
                return None
            aspect = self.width / self.height
            if x / y >= aspect:
                x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
            else:
                y = round_aspect(x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
            return (x, y)
        box = None
        if reducing_gap is not None:
            size = preserve_aspect_ratio()
            if size is None:
                return
            res = self.draft(None, (size[0] * reducing_gap, size[1] * reducing_gap))
            if res is not None:
                box = res[1]
        if box is None:
            self.load()
            size = preserve_aspect_ratio()
            if size is None:
                return
        if self.size != size:
            im = self.resize(size, resample, box=box, reducing_gap=reducing_gap)
            self.im = im.im
            self._size = size
            self._mode = self.im.mode
        self.readonly = 0
        self.pyaccess = None
