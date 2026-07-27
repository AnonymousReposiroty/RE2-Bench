import re
import warnings

def _parse_glibc_version(version_str: str) -> _GLibCVersion:
    """Parse glibc version.

    We use a regexp instead of str.split because we want to discard any
    random junk that might come after the minor version -- this might happen
    in patched/forked versions of glibc (e.g. Linaro's version of glibc
    uses version strings like "2.20-2014.11"). See gh-3588.
    """
    m = re.match(r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)", version_str)
    if not m:
        warnings.warn(
            f"Expected glibc version with 2 components major.minor, got: {version_str}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _GLibCVersion(-1, -1)
    return _GLibCVersion(int(m.group("major")), int(m.group("minor")))
