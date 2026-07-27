import sys
from typing import Any, Literal, SupportsFloat, SupportsInt, Union
from . import _imagingcms as core
from . import ImageWin  # type: ignore[unused-ignore, unreachable]

def get_display_profile(handle: SupportsInt | None = None) -> ImageCmsProfile | None:
    """
    (experimental) Fetches the profile for the current display device.

    :returns: ``None`` if the profile is not known.
    """

    if sys.platform != "win32":
        return None

    from . import ImageWin  # type: ignore[unused-ignore, unreachable]

    if isinstance(handle, ImageWin.HDC):
        profile = core.get_display_profile_win32(int(handle), 1)
    else:
        profile = core.get_display_profile_win32(int(handle or 0))
    if profile is None:
        return None
    return ImageCmsProfile(profile)
