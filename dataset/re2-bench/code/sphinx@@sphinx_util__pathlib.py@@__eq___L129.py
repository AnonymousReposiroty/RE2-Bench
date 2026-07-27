import warnings
from pathlib import Path, PosixPath, PurePath, WindowsPath
from sphinx.deprecation import RemovedInSphinx10Warning

class _StrPath(PosixPath):

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PurePath):
            return super().__eq__(other)
        if isinstance(other, str):
            warnings.warn(_MSG, RemovedInSphinx10Warning, stacklevel=2)
            return self.__str__() == other
        return NotImplemented
