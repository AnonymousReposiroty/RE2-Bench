import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale, _image

class BivarColormap:
    """
    Base class for all bivariate to RGBA mappings.

    Designed as a drop-in replacement for Colormap when using a 2D
    lookup table. To be used with `~matplotlib.cm.ScalarMappable`.
    """

    def __init__(self, N=256, M=256, shape='square', origin=(0, 0), name='bivariate colormap'):
        """
        Parameters
        ----------
        N : int, default: 256
            The number of RGB quantization levels along the first axis.
        M : int, default: 256
            The number of RGB quantization levels along the second axis.
        shape : {'square', 'circle', 'ignore', 'circleignore'}

            - 'square' each variate is clipped to [0,1] independently
            - 'circle' the variates are clipped radially to the center
              of the colormap, and a circular mask is applied when the colormap
              is displayed
            - 'ignore' the variates are not clipped, but instead assigned the
              'outside' color
            - 'circleignore' a circular mask is applied, but the data is not
              clipped and instead assigned the 'outside' color

        origin : (float, float), default: (0,0)
            The relative origin of the colormap. Typically (0, 0), for colormaps
            that are linear on both axis, and (.5, .5) for circular colormaps.
            Used when getting 1D colormaps from 2D colormaps.
        name : str, optional
            The name of the colormap.
        """
        self.name = name
        self.N = int(N)
        self.M = int(M)
        _api.check_in_list(['square', 'circle', 'ignore', 'circleignore'], shape=shape)
        self._shape = shape
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0)
        self._rgba_outside = (1.0, 0.0, 1.0, 1.0)
        self._isinit = False
        self.n_variates = 2
        self._origin = (float(origin[0]), float(origin[1]))
        '#: When this colormap exists on a scalar mappable and colorbar_extend\n        #: is not False, colorbar creation will pick up ``colorbar_extend`` as\n        #: the default value for the ``extend`` keyword in the\n        #: `matplotlib.colorbar.Colorbar` constructor.\n        self.colorbar_extend = False'

    @property
    def lut(self):
        """
        For external access to the lut, i.e. for displaying the cmap.
        For circular colormaps this returns a lut with a circular mask.

        Internal functions (such as to_rgb()) should use _lut
        which stores the lut without a circular mask
        A lut without the circular mask is needed in to_rgb() because the
        conversion from floats to ints results in some some pixel-requests
        just outside of the circular mask

        """
        if not self._isinit:
            self._init()
        lut = np.copy(self._lut)
        if self.shape == 'circle' or self.shape == 'circleignore':
            n = np.linspace(-1, 1, self.N)
            m = np.linspace(-1, 1, self.M)
            radii_sqr = (n ** 2)[:, np.newaxis] + (m ** 2)[np.newaxis, :]
            mask_outside = radii_sqr > 1
            lut[mask_outside, 3] = 0
        return lut

    def _init(self):
        """Generate the lookup table, ``self._lut``."""
        raise NotImplementedError('Abstract class only')

    @property
    def shape(self):
        return self._shape

    def _repr_png_(self):
        """Generate a PNG representation of the BivarColormap."""
        if not self._isinit:
            self._init()
        pixels = self.lut
        if pixels.shape[0] < _BIVAR_REPR_PNG_SIZE:
            pixels = np.repeat(pixels, repeats=_BIVAR_REPR_PNG_SIZE // pixels.shape[0], axis=0)[:256, :]
        if pixels.shape[1] < _BIVAR_REPR_PNG_SIZE:
            pixels = np.repeat(pixels, repeats=_BIVAR_REPR_PNG_SIZE // pixels.shape[1], axis=1)[:, :256]
        pixels = (pixels[::-1, :, :] * 255).astype(np.uint8)
        png_bytes = io.BytesIO()
        title = self.name + ' BivarColormap'
        author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        pnginfo.add_text('Author', author)
        pnginfo.add_text('Software', author)
        Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()
