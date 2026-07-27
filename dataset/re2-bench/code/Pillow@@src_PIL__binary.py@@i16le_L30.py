from struct import pack, unpack_from

def i16le(c: bytes, o: int = 0) -> int:
    """
    Converts a 2-bytes (16 bits) string to an unsigned integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<H", c, o)[0]
