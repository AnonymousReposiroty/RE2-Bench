import numpy as np

def safe_masked_invalid(x, copy=False):
    x = np.array(x, subok=True, copy=copy)
    if not x.dtype.isnative:
        # If we have already made a copy, do the byteswap in place, else make a
        # copy with the byte order swapped.
        # Swap to native order.
        x = x.byteswap(inplace=copy).view(x.dtype.newbyteorder('N'))
    try:
        xm = np.ma.masked_where(~(np.isfinite(x)), x, copy=False)
    except TypeError:
        if len(x.dtype.descr) == 1:
            # Arrays with dtype 'object' get returned here.
            # For example the 'c' kwarg of scatter, which supports multiple types.
            # `plt.scatter([3, 4], [2, 5], c=[(1, 0, 0), 'y'])`
            return x
        else:
            # In case of a dtype with multiple fields
            # for example image data using a MultiNorm
            try:
                mask = np.empty(x.shape, dtype=np.dtype('bool, '*len(x.dtype.descr)))
                for dd, dm in zip(x.dtype.descr, mask.dtype.descr):
                    mask[dm[0]] = ~np.isfinite(x[dd[0]])
                xm = np.ma.array(x, mask=mask, copy=False)
            except TypeError:
                return x
    return xm
