import sys
from . import Image, __version__
from ._deprecate import deprecate
from . import _imagingcms as core

def versions() -> tuple[str, str, str, str]:
    """
    (pyCMS) Fetches versions.
    """

    deprecate(
        "PIL.ImageCms.versions()",
        12,
        '(PIL.features.version("littlecms2"), sys.version, PIL.__version__)',
    )
    return _VERSION, core.littlecms_version, sys.version.split()[0], __version__
