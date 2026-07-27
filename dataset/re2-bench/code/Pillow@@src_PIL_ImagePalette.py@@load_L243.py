from . import GimpGradientFile, GimpPaletteFile, ImageColor, PaletteFile

def load(filename):
    # FIXME: supports GIMP gradients only

    with open(filename, "rb") as fp:
        for paletteHandler in [
            GimpPaletteFile.GimpPaletteFile,
            GimpGradientFile.GimpGradientFile,
            PaletteFile.PaletteFile,
        ]:
            try:
                fp.seek(0)
                lut = paletteHandler(fp).getpalette()
                if lut:
                    break
            except (SyntaxError, ValueError):
                pass
        else:
            msg = "cannot load palette"
            raise OSError(msg)

    return lut  # data, rawmode
