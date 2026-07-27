def bit_scan1(x, n=0):
    if not x:
        return
    x = abs(x >> n)
    low_byte = x & 0xFF
    if low_byte:
        return _small_trailing[low_byte] + n

    t = 8 + n
    x >>= 8
    # 2**m is quick for z up through 2**30
    z = x.bit_length() - 1
    if x == 1 << z:
        return z + t

    if z < 300:
        # fixed 8-byte reduction
        while not x & 0xFF:
            x >>= 8
            t += 8
    else:
        # binary reduction important when there might be a large
        # number of trailing 0s
        p = z >> 1
        while not x & 0xFF:
            while x & ((1 << p) - 1):
                p >>= 1
            x >>= p
            t += p
    return t + _small_trailing[x & 0xFF]
