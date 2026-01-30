class PdfDict(_DictBase):

    def __bytes__(self):
        out = bytearray(b'<<')
        for key, value in self.items():
            if value is None:
                continue
            value = pdf_repr(value)
            out.extend(b'\n')
            out.extend(bytes(PdfName(key)))
            out.extend(b' ')
            out.extend(value)
        out.extend(b'\n>>')
        return bytes(out)
