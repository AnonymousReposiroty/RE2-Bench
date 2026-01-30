import datetime
import dateutil.parser
import dateutil.tz
import matplotlib as mpl

def _get_tzinfo(tz=None):
    """
    Generate `~datetime.tzinfo` from a string or return `~datetime.tzinfo`.
    If None, retrieve the preferred timezone from the rcParams dictionary.
    """
    tz = mpl._val_or_rc(tz, 'timezone')
    if tz == 'UTC':
        return UTC
    if isinstance(tz, str):
        tzinfo = dateutil.tz.gettz(tz)
        if tzinfo is None:
            raise ValueError(f"{tz} is not a valid timezone as parsed by"
                             " dateutil.tz.gettz.")
        return tzinfo
    if isinstance(tz, datetime.tzinfo):
        return tz
    raise TypeError(f"tz must be string or tzinfo subclass, not {tz!r}.")
