import warnings
from typing import TYPE_CHECKING, Any, Literal

class _ExtraMeta(type):

    def __getattribute__(self, __name: str) -> Any:
        if __name in {'allow', 'ignore', 'forbid'}:
            warnings.warn("`pydantic.config.Extra` is deprecated, use literal values instead (e.g. `extra='allow'`)", DeprecationWarning, stacklevel=2)
        return super().__getattribute__(__name)
