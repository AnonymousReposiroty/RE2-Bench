import numpy as np
from matplotlib import _api, cbook, ticker, units

def date2num(d):
    """
    Convert datetime objects to Matplotlib dates.

    Parameters
    ----------
    d : `datetime.datetime` or `numpy.datetime64` or sequences of these

    Returns
    -------
    float or sequence of floats
        Number of days since the epoch.  See `.get_epoch` for the
        epoch, which can be changed by :rc:`date.epoch` or `.set_epoch`.  If
        the epoch is "1970-01-01T00:00:00" (default) then noon Jan 1 1970
        ("1970-01-01T12:00:00") returns 0.5.

    Notes
    -----
    The Gregorian calendar is assumed; this is not universal practice.
    For details see the module docstring.
    """
    # Unpack in case of e.g. Pandas or xarray object
    d = cbook._unpack_to_numpy(d)

    # make an iterable, but save state to unpack later:
    iterable = np.iterable(d)
    if not iterable:
        d = [d]

    masked = np.ma.is_masked(d)
    mask = np.ma.getmask(d)
    d = np.asarray(d)

    # convert to datetime64 arrays, if not already:
    if not np.issubdtype(d.dtype, np.datetime64):
        # datetime arrays
        if not d.size:
            # deals with an empty array...
            return d
        tzi = getattr(d[0], 'tzinfo', None)
        if tzi is not None:
            # make datetime naive:
            d = [dt.astimezone(UTC).replace(tzinfo=None) for dt in d]
            d = np.asarray(d)
        d = d.astype('datetime64[us]')

    d = np.ma.masked_array(d, mask=mask) if masked else d
    d = _dt64_to_ordinalf(d)

    return d if iterable else d[0]
