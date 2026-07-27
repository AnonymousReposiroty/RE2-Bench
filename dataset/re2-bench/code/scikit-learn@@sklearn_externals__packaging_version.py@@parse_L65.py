from typing import Callable, Iterator, List, Optional, SupportsInt, Tuple, Union

def parse(version: str) -> Union["LegacyVersion", "Version"]:
    """Parse the given version from a string to an appropriate class.

    Parameters
    ----------
    version : str
        Version in a string format, eg. "0.9.1" or "1.2.dev0".

    Returns
    -------
    version : :class:`Version` object or a :class:`LegacyVersion` object
        Returned class depends on the given version: if is a valid
        PEP 440 version or a legacy version.
    """
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)
