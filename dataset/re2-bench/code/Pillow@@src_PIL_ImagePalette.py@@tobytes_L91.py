import array
from typing import Sequence

class ImagePalette:
    """
    Color palette for palette mapped images

    :param mode: The mode to use for the palette. See:
        :ref:`concept-modes`. Defaults to "RGB"
    :param palette: An optional palette. If given, it must be a bytearray,
        an array or a list of ints between 0-255. The list must consist of
        all channels for one color followed by the next color (e.g. RGBRGBRGB).
        Defaults to an empty palette.
    """

    def __init__(self, mode: str='RGB', palette: Sequence[int] | None=None) -> None:
        self.mode = mode
        self.rawmode = None
        self.palette = palette or bytearray()
        self.dirty: int | None = None

    @property
    def palette(self):
        return self._palette

    @palette.setter
    def palette(self, palette):
        self._colors = None
        self._palette = palette

    def tobytes(self):
        """Convert palette to bytes.

        .. warning:: This method is experimental.
        """
        if self.rawmode:
            msg = 'palette contains raw palette data'
            raise ValueError(msg)
        if isinstance(self.palette, bytes):
            return self.palette
        arr = array.array('B', self.palette)
        return arr.tobytes()
    tostring = tobytes
