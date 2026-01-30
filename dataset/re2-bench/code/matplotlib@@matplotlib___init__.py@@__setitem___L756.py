from collections.abc import MutableMapping
from . import _api, _version, cbook, _docstring, rcsetup

@_docstring.Substitution('\n'.join(map('- {}'.format, sorted(rcsetup._validators, key=str.lower))))
class RcParams(MutableMapping, dict):
    """
    A dict-like key-value store for config parameters, including validation.

    Validating functions are defined and associated with rc parameters in
    :mod:`matplotlib.rcsetup`.

    The list of rcParams is:

    %s

    See Also
    --------
    :ref:`customizing-with-matplotlibrc-files`
    """
    validate = rcsetup._validators

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def _set(self, key, val):
        """
        Directly write data bypassing deprecation and validation logic.

        Notes
        -----
        As end user or downstream library you almost always should use
        ``rcParams[key] = val`` and not ``_set()``.

        There are only very few special cases that need direct data access.
        These cases previously used ``dict.__setitem__(rcParams, key, val)``,
        which is now deprecated and replaced by ``rcParams._set(key, val)``.

        Even though private, we guarantee API stability for ``rcParams._set``,
        i.e. it is subject to Matplotlib's API and deprecation policy.

        :meta public:
        """
        dict.__setitem__(self, key, val)

    def __setitem__(self, key, val):
        if key == 'backend' and val is rcsetup._auto_backend_sentinel and ('backend' in self):
            return
        valid_key = _api.check_getitem(self.validate, rcParam=key, _error_cls=KeyError)
        try:
            cval = valid_key(val)
        except ValueError as ve:
            raise ValueError(f'Key {key}: {ve}') from None
        self._set(key, cval)
