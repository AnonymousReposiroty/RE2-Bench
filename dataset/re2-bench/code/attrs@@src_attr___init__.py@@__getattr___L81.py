from typing import Callable, Literal, Protocol
from ._version_info import VersionInfo
from importlib.metadata import metadata

def _make_getattr(mod_name: str) -> Callable:
    """
    Create a metadata proxy for packaging information that uses *mod_name* in
    its warnings and errors.
    """

    def __getattr__(name: str) -> str:
        if name not in ("__version__", "__version_info__"):
            msg = f"module {mod_name} has no attribute {name}"
            raise AttributeError(msg)

        from importlib.metadata import metadata

        meta = metadata("attrs")

        if name == "__version_info__":
            return VersionInfo._from_version_string(meta["version"])

        return meta["version"]

    return __getattr__
