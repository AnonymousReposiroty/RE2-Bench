import io
from . import Image

class Parser:
    """
    Incremental image parser.  This class implements the standard
    feed/close consumer interface.
    """
    incremental = None
    image: Image.Image | None = None
    data = None
    decoder = None
    offset = 0
    finished = 0

    def feed(self, data):
        """
        (Consumer) Feed data to the parser.

        :param data: A string buffer.
        :exception OSError: If the parser failed to parse the image file.
        """
        if self.finished:
            return
        if self.data is None:
            self.data = data
        else:
            self.data = self.data + data
        if self.decoder:
            if self.offset > 0:
                skip = min(len(self.data), self.offset)
                self.data = self.data[skip:]
                self.offset = self.offset - skip
                if self.offset > 0 or not self.data:
                    return
            n, e = self.decoder.decode(self.data)
            if n < 0:
                self.data = None
                self.finished = 1
                if e < 0:
                    self.image = None
                    raise _get_oserror(e, encoder=False)
                else:
                    return
            self.data = self.data[n:]
        elif self.image:
            pass
        else:
            try:
                with io.BytesIO(self.data) as fp:
                    im = Image.open(fp)
            except OSError:
                pass
            else:
                flag = hasattr(im, 'load_seek') or hasattr(im, 'load_read')
                if flag or len(im.tile) != 1:
                    self.decode = None
                else:
                    im.load_prepare()
                    d, e, o, a = im.tile[0]
                    im.tile = []
                    self.decoder = Image._getdecoder(im.mode, d, a, im.decoderconfig)
                    self.decoder.setimage(im.im, e)
                    self.offset = o
                    if self.offset <= len(self.data):
                        self.data = self.data[self.offset:]
                        self.offset = 0
                self.image = im
