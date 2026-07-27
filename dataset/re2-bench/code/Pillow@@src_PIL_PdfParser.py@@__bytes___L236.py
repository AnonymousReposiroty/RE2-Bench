class PdfName:

    def __init__(self, name):
        if isinstance(name, PdfName):
            self.name = name.name
        elif isinstance(name, bytes):
            self.name = name
        else:
            self.name = name.encode('us-ascii')
    allowed_chars = set(range(33, 127)) - {ord(c) for c in '#%/()<>[]{}'}

    def __bytes__(self):
        result = bytearray(b'/')
        for b in self.name:
            if b in self.allowed_chars:
                result.append(b)
            else:
                result.extend(b'#%02X' % b)
        return bytes(result)
