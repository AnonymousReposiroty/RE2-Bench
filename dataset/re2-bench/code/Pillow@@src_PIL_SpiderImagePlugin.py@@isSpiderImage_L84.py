import struct

def isSpiderImage(filename):
    with open(filename, "rb") as fp:
        f = fp.read(92)  # read 23 * 4 bytes
    t = struct.unpack(">23f", f)  # try big-endian first
    hdrlen = isSpiderHeader(t)
    if hdrlen == 0:
        t = struct.unpack("<23f", f)  # little-endian
        hdrlen = isSpiderHeader(t)
    return hdrlen
