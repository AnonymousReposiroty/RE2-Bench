from collections.abc import MutableMapping
import re
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

    def find_all(self, pattern):
        """
        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        .. note::

            Changes to the returned dictionary are *not* propagated to
            the parent RcParams dictionary.

        """
        pattern_re = re.compile(pattern)
        return self.__class__(((key, value) for key, value in self.items() if pattern_re.search(key)))
