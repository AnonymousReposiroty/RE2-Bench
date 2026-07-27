import time

def pdf_repr(x):
    if x is True:
        return b"true"
    elif x is False:
        return b"false"
    elif x is None:
        return b"null"
    elif isinstance(x, (PdfName, PdfDict, PdfArray, PdfBinary)):
        return bytes(x)
    elif isinstance(x, (int, float)):
        return str(x).encode("us-ascii")
    elif isinstance(x, time.struct_time):
        return b"(D:" + time.strftime("%Y%m%d%H%M%SZ", x).encode("us-ascii") + b")"
    elif isinstance(x, dict):
        return bytes(PdfDict(x))
    elif isinstance(x, list):
        return bytes(PdfArray(x))
    elif isinstance(x, str):
        return pdf_repr(encode_text(x))
    elif isinstance(x, bytes):
        # XXX escape more chars? handle binary garbage
        x = x.replace(b"\\", b"\\\\")
        x = x.replace(b"(", b"\\(")
        x = x.replace(b")", b"\\)")
        return b"(" + x + b")"
    else:
        return bytes(x)
