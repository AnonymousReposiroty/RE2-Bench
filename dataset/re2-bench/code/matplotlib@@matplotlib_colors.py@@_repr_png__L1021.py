import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np

class Colormap:
    """
    Baseclass for all scalar to RGBA mappings.

    Typically, Colormap instances are used to convert data values (floats)
    from the interval ``[0, 1]`` to the RGBA color that the respective
    Colormap represents. For scaling of data into the ``[0, 1]`` interval see
    `matplotlib.colors.Normalize`. Subclasses of `matplotlib.cm.ScalarMappable`
    make heavy use of this ``data -> normalize -> map-to-color`` processing
    chain.
    """

    def __init__(self, name, N=256, *, bad=None, under=None, over=None):
        """
        Parameters
        ----------
        name : str
            The name of the colormap.
        N : int
            The number of RGB quantization levels.
        bad : :mpltype:`color`, default: transparent
            The color for invalid values (NaN or masked).

            .. versionadded:: 3.11

        under : :mpltype:`color`, default: color of the lowest value
            The color for low out-of-range values.

            .. versionadded:: 3.11

        over : :mpltype:`color`, default: color of the highest value
            The color for high out-of-range values.

            .. versionadded:: 3.11
        """
        self.name = name
        self.N = int(N)
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0) if bad is None else to_rgba(bad)
        self._rgba_under = None if under is None else to_rgba(under)
        self._rgba_over = None if over is None else to_rgba(over)
        self._i_under = self.N
        self._i_over = self.N + 1
        self._i_bad = self.N + 2
        self._isinit = False
        self.n_variates = 1
        self.colorbar_extend = False

    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        X = np.tile(np.linspace(0, 1, _REPR_PNG_SIZE[0]), (_REPR_PNG_SIZE[1], 1))
        pixels = self(X, bytes=True)
        png_bytes = io.BytesIO()
        title = self.name + ' colormap'
        author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        pnginfo.add_text('Author', author)
        pnginfo.add_text('Software', author)
        Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()
