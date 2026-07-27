from ._internal import _config, _decorators, _mock_val_ser, _namespace_utils, _typing_extra
from ._internal import _dataclasses as _pydantic_dataclasses
from ._internal._dataclasses import PydanticDataclass
from ._internal._namespace_utils import MappingNamespace

def rebuild_dataclass(
    cls: type[PydanticDataclass],
    *,
    force: bool = False,
    raise_errors: bool = True,
    _parent_namespace_depth: int = 2,
    _types_namespace: MappingNamespace | None = None,
) -> bool | None:
    """Try to rebuild the pydantic-core schema for the dataclass.

    This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
    the initial attempt to build the schema, and automatic rebuilding fails.

    This is analogous to `BaseModel.model_rebuild`.

    Args:
        cls: The class to rebuild the pydantic-core schema for.
        force: Whether to force the rebuilding of the schema, defaults to `False`.
        raise_errors: Whether to raise errors, defaults to `True`.
        _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
        _types_namespace: The types namespace, defaults to `None`.

    Returns:
        Returns `None` if the schema is already "complete" and rebuilding was not required.
        If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.
    """
    if not force and cls.__pydantic_complete__:
        return None

    for attr in ('__pydantic_core_schema__', '__pydantic_validator__', '__pydantic_serializer__'):
        if attr in cls.__dict__ and not isinstance(getattr(cls, attr), _mock_val_ser.MockValSer):
            # Deleting the validator/serializer is necessary as otherwise they can get reused in
            # pydantic-core. Same applies for the core schema that can be reused in schema generation.
            delattr(cls, attr)

    cls.__pydantic_complete__ = False

    if _types_namespace is not None:
        rebuild_ns = _types_namespace
    elif _parent_namespace_depth > 0:
        rebuild_ns = _typing_extra.parent_frame_namespace(parent_depth=_parent_namespace_depth, force=True) or {}
    else:
        rebuild_ns = {}

    ns_resolver = _namespace_utils.NsResolver(
        parent_namespace=rebuild_ns,
    )

    return _pydantic_dataclasses.complete_dataclass(
        cls,
        _config.ConfigWrapper(cls.__pydantic_config__, check=False),
        raise_errors=raise_errors,
        ns_resolver=ns_resolver,
        # We could provide a different config instead (with `'defer_build'` set to `True`)
        # of this explicit `_force_build` argument, but because config can come from the
        # decorator parameter or the `__pydantic_config__` attribute, `complete_dataclass`
        # will overwrite `__pydantic_config__` with the provided config above:
        _force_build=True,
    )
