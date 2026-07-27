import datetime
import functools
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
                            MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY,
                            SECONDLY)

class rrulewrapper:
    """
    A simple wrapper around a `dateutil.rrule` allowing flexible
    date tick specifications.
    """

    def __init__(self, freq, tzinfo=None, **kwargs):
        """
        Parameters
        ----------
        freq : {YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY}
            Tick frequency. These constants are defined in `dateutil.rrule`,
            but they are accessible from `matplotlib.dates` as well.
        tzinfo : `datetime.tzinfo`, optional
            Time zone information. The default is None.
        **kwargs
            Additional keyword arguments are passed to the `dateutil.rrule`.
        """
        kwargs['freq'] = freq
        self._base_tzinfo = tzinfo
        self._update_rrule(**kwargs)

    def _update_rrule(self, **kwargs):
        tzinfo = self._base_tzinfo
        if 'dtstart' in kwargs:
            dtstart = kwargs['dtstart']
            if dtstart.tzinfo is not None:
                if tzinfo is None:
                    tzinfo = dtstart.tzinfo
                else:
                    dtstart = dtstart.astimezone(tzinfo)
                kwargs['dtstart'] = dtstart.replace(tzinfo=None)
        if 'until' in kwargs:
            until = kwargs['until']
            if until.tzinfo is not None:
                if tzinfo is not None:
                    until = until.astimezone(tzinfo)
                else:
                    raise ValueError('until cannot be aware if dtstart is naive and tzinfo is None')
                kwargs['until'] = until.replace(tzinfo=None)
        self._construct = kwargs.copy()
        self._tzinfo = tzinfo
        self._rrule = rrule(**self._construct)

    def _attach_tzinfo(self, dt, tzinfo):
        if hasattr(tzinfo, 'localize'):
            return tzinfo.localize(dt, is_dst=True)
        return dt.replace(tzinfo=tzinfo)

    def _aware_return_wrapper(self, f, returns_list=False):
        """Decorator function that allows rrule methods to handle tzinfo."""
        if self._tzinfo is None:
            return f

        def normalize_arg(arg):
            if isinstance(arg, datetime.datetime) and arg.tzinfo is not None:
                if arg.tzinfo is not self._tzinfo:
                    arg = arg.astimezone(self._tzinfo)
                return arg.replace(tzinfo=None)
            return arg

        def normalize_args(args, kwargs):
            args = tuple((normalize_arg(arg) for arg in args))
            kwargs = {kw: normalize_arg(arg) for kw, arg in kwargs.items()}
            return (args, kwargs)
        if not returns_list:

            def inner_func(*args, **kwargs):
                args, kwargs = normalize_args(args, kwargs)
                dt = f(*args, **kwargs)
                return self._attach_tzinfo(dt, self._tzinfo)
        else:

            def inner_func(*args, **kwargs):
                args, kwargs = normalize_args(args, kwargs)
                dts = f(*args, **kwargs)
                return [self._attach_tzinfo(dt, self._tzinfo) for dt in dts]
        return functools.wraps(f)(inner_func)
