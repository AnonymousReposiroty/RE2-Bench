import dataclasses as _dataclasses
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
)
from pydantic_core import CoreSchema, PydanticCustomError, SchemaSerializer, core_schema
from ._internal import _fields, _internal_dataclass, _utils, _validators
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .json_schema import JsonSchemaValue

@_dataclasses.dataclass(**_internal_dataclass.slots_true)
class UuidVersion:
    """A field metadata class to indicate a [UUID](https://docs.python.org/3/library/uuid.html) version.

    Use this class as an annotation via [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated), as seen below.

    Attributes:
        uuid_version: The version of the UUID. Must be one of 1, 3, 4, 5, 6, 7 or 8.

    Example:
        ```python
        from typing import Annotated
        from uuid import UUID

        from pydantic.types import UuidVersion

        UUID1 = Annotated[UUID, UuidVersion(1)]
        ```
    """

    uuid_version: Literal[1, 3, 4, 5, 6, 7, 8]

    def __get_pydantic_json_schema__(
        self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        field_schema = handler(core_schema)
        field_schema.pop('anyOf', None)  # remove the bytes/str union
        field_schema.update(type='string', format=f'uuid{self.uuid_version}')
        return field_schema

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        schema = handler(source)
        _check_annotated_type(schema['type'], 'uuid', self.__class__.__name__)
        schema['version'] = self.uuid_version  # type: ignore
        return schema

    def __hash__(self) -> int:
        return hash(type(self.uuid_version))
