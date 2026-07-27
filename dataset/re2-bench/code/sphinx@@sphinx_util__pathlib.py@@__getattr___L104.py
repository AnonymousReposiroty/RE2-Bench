import warnings
from pathlib import Path, PosixPath, PurePath, WindowsPath
from sphinx.deprecation import RemovedInSphinx10Warning
from typing import Any

class _StrPath(PosixPath):

    def __getattr__(self, item: str) -> Any:
        if item in _STR_METHODS:
            warnings.warn(_MSG, RemovedInSphinx10Warning, stacklevel=2)
            return getattr(self.__str__(), item)
        msg = f'{_PATH_NAME!r} has no attribute {item!r}'
        raise AttributeError(msg)
