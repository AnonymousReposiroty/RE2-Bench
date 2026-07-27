from . import Image, ImageFile, ImagePalette

class PsdImageFile(ImageFile.ImageFile):
    format = 'PSD'
    format_description = 'Adobe Photoshop'
    _close_exclusive_fp_after_loading = False

    def seek(self, layer):
        if not self._seek_check(layer):
            return
        try:
            name, mode, bbox, tile = self.layers[layer - 1]
            self._mode = mode
            self.tile = tile
            self.frame = layer
            self.fp = self._fp
            return (name, bbox)
        except IndexError as e:
            msg = 'no such layer'
            raise EOFError(msg) from e
