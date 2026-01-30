import warnings
from typing import TYPE_CHECKING, Any, Literal
from .._internal import _config

class _ConfigMetaclass(type):
    def __getattr__(self, item: str) -> Any:
        try:
            obj = _config.config_defaults[item]
            warnings.warn(_config.DEPRECATION_MESSAGE, DeprecationWarning)
            return obj
        except KeyError as exc:
            raise AttributeError(f"type object '{self.__name__}' has no attribute {exc}") from exc
