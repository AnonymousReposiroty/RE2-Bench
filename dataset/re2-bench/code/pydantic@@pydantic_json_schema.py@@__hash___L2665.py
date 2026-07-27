import dataclasses
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    NewType,
    TypeVar,
    Union,
    cast,
    overload,
)
from pydantic_core import MISSING, CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from ._internal import (
    _config,
    _core_metadata,
    _core_utils,
    _decorators,
    _internal_dataclass,
    _mock_val_ser,
    _schema_generation_shared,
)
from .annotated_handlers import GetJsonSchemaHandler

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class WithJsonSchema:
    """!!! abstract "Usage Documentation"
        [`WithJsonSchema` Annotation](../concepts/json_schema.md#withjsonschema-annotation)

    Add this as an annotation on a field to override the (base) JSON schema that would be generated for that field.
    This provides a way to set a JSON schema for types that would otherwise raise errors when producing a JSON schema,
    such as Callable, or types that have an is-instance core schema, without needing to go so far as creating a
    custom subclass of pydantic.json_schema.GenerateJsonSchema.
    Note that any _modifications_ to the schema that would normally be made (such as setting the title for model fields)
    will still be performed.

    If `mode` is set this will only apply to that schema generation mode, allowing you
    to set different json schemas for validation and serialization.
    """

    json_schema: JsonSchemaValue | None
    mode: Literal['validation', 'serialization'] | None = None

    def __get_pydantic_json_schema__(
        self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        mode = self.mode or handler.mode
        if mode != handler.mode:
            return handler(core_schema)
        if self.json_schema is None:
            # This exception is handled in pydantic.json_schema.GenerateJsonSchema._named_required_fields_schema
            raise PydanticOmit
        else:
            return self.json_schema.copy()

    def __hash__(self) -> int:
        return hash(type(self.mode))
