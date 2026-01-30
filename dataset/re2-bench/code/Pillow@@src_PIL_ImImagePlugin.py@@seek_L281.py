from . import Image, ImageFile, ImagePalette

class ImImageFile(ImageFile.ImageFile):
    format = 'IM'
    format_description = 'IFUNC Image Memory'
    _close_exclusive_fp_after_loading = False

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        self.frame = frame
        if self.mode == '1':
            bits = 1
        else:
            bits = 8 * len(self.mode)
        size = (self.size[0] * bits + 7) // 8 * self.size[1]
        offs = self.__offset + frame * size
        self.fp = self._fp
        self.tile = [('raw', (0, 0) + self.size, offs, (self.rawmode, 0, -1))]
