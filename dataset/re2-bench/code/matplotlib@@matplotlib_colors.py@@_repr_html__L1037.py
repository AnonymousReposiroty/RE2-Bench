import base64
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

    def get_bad(self):
        """Get the color for masked values."""
        self._ensure_inited()
        return np.array(self._lut[self._i_bad])

    def get_under(self):
        """Get the color for low out-of-range values."""
        self._ensure_inited()
        return np.array(self._lut[self._i_under])

    def get_over(self):
        """Get the color for high out-of-range values."""
        self._ensure_inited()
        return np.array(self._lut[self._i_over])

    def _init(self):
        """Generate the lookup table, ``self._lut``."""
        raise NotImplementedError('Abstract class only')

    def _ensure_inited(self):
        if not self._isinit:
            self._init()

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

    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        png_bytes = self._repr_png_()
        png_base64 = base64.b64encode(png_bytes).decode('ascii')

        def color_block(color):
            hex_color = to_hex(color, keep_alpha=True)
            return f'<div title="{hex_color}" style="display: inline-block; width: 1em; height: 1em; margin: 0; vertical-align: middle; border: 1px solid #555; background-color: {hex_color};"></div>'
        return f'<div style="vertical-align: middle;"><strong>{self.name}</strong> </div><div class="cmap"><img alt="{self.name} colormap" title="{self.name}" style="border: 1px solid #555;" src="data:image/png;base64,{png_base64}"></div><div style="vertical-align: middle; max-width: {_REPR_PNG_SIZE[0] + 2}px; display: flex; justify-content: space-between;"><div style="float: left;">{color_block(self.get_under())} under</div><div style="margin: 0 auto; display: inline-block;">bad {color_block(self.get_bad())}</div><div style="float: right;">over {color_block(self.get_over())}</div></div>'
