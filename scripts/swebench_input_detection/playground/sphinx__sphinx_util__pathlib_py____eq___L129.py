# Problem: sphinx@@sphinx_util__pathlib.py@@__eq___L129
# Module: sphinx.util._pathlib
# Function: __eq__
# Line: 129

import warnings
from pathlib import Path, PosixPath, PurePath, WindowsPath
from sphinx.deprecation import RemovedInSphinx80Warning
_MSG = (
    'Sphinx 10 will drop support for representing paths as strings. '
    'Use "pathlib.Path" or "os.fspath" instead.'
)

class _StrPath(PosixPath):

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PurePath):
            return super().__eq__(other)
        if isinstance(other, str):
            warnings.warn(_MSG, RemovedInSphinx80Warning, stacklevel=2)
            return self.__str__() == other
        return NotImplemented


def test_input(pred_input):
    obj_ins = _StrPath()
    obj_ins_pred = _StrPath()
    assert obj_ins.__eq__(other = _StrPath('/tmp/pytest-of-changshu/pytest-41/root/_build/html'))==obj_ins_pred.__eq__(other = pred_input['args']['other']), 'Prediction failed!'