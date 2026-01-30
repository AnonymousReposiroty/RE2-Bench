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

    def _get(self, key):
        """
        Directly read data bypassing deprecation, backend and validation
        logic.

        Notes
        -----
        As end user or downstream library you almost always should use
        ``val = rcParams[key]`` and not ``_get()``.

        There are only very few special cases that need direct data access.
        These cases previously used ``dict.__getitem__(rcParams, key, val)``,
        which is now deprecated and replaced by ``rcParams._get(key)``.

        Even though private, we guarantee API stability for ``rcParams._get``,
        i.e. it is subject to Matplotlib's API and deprecation policy.

        :meta public:
        """
        return dict.__getitem__(self, key)

    def copy(self):
        """Copy this RcParams instance."""
        rccopy = self.__class__()
        for k in self:
            rccopy._set(k, self._get(k))
        return rccopy
