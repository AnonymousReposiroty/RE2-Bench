import operator
import warnings
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import Self, TypeAlias, Unpack
from ._internal import (
    _config,
    _decorators,
    _fields,
    _forward_ref,
    _generics,
    _mock_val_ser,
    _model_construction,
    _namespace_utils,
    _repr,
    _typing_extra,
    _utils,
)
from .config import ConfigDict, ExtraValues
from .plugin._schema_validator import PluggableSchemaValidator
from inspect import Signature
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator
from .fields import ComputedFieldInfo, FieldInfo, ModelPrivateAttr

class BaseModel(metaclass=_model_construction.ModelMetaclass):
    """!!! abstract "Usage Documentation"
        [Models](../concepts/models.md)

    A base class for creating Pydantic models.

    Attributes:
        __class_vars__: The names of the class variables defined on the model.
        __private_attributes__: Metadata about the private attributes of the model.
        __signature__: The synthesized `__init__` [`Signature`][inspect.Signature] of the model.

        __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
        __pydantic_core_schema__: The core schema of the model.
        __pydantic_custom_init__: Whether the model has a custom `__init__` function.
        __pydantic_decorators__: Metadata containing the decorators defined on the model.
            This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
        __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
            __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
        __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
        __pydantic_post_init__: The name of the post-init method for the model, if defined.
        __pydantic_root_model__: Whether the model is a [`RootModel`][pydantic.root_model.RootModel].
        __pydantic_serializer__: The `pydantic-core` `SchemaSerializer` used to dump instances of the model.
        __pydantic_validator__: The `pydantic-core` `SchemaValidator` used to validate instances of the model.

        __pydantic_fields__: A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.
        __pydantic_computed_fields__: A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.

        __pydantic_extra__: A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra]
            is set to `'allow'`.
        __pydantic_fields_set__: The names of fields explicitly set during instantiation.
        __pydantic_private__: Values of private attributes set on the model instance.
    """
    model_config: ClassVar[ConfigDict] = ConfigDict()
    '\n    Configuration for the model, should be a dictionary conforming to [`ConfigDict`][pydantic.config.ConfigDict].\n    '
    __class_vars__: ClassVar[set[str]]
    'The names of the class variables defined on the model.'
    __private_attributes__: ClassVar[Dict[str, ModelPrivateAttr]]
    'Metadata about the private attributes of the model.'
    __signature__: ClassVar[Signature]
    'The synthesized `__init__` [`Signature`][inspect.Signature] of the model.'
    __pydantic_complete__: ClassVar[bool] = False
    'Whether model building is completed, or if there are still undefined fields.'
    __pydantic_core_schema__: ClassVar[CoreSchema]
    'The core schema of the model.'
    __pydantic_custom_init__: ClassVar[bool]
    'Whether the model has a custom `__init__` method.'
    __pydantic_decorators__: ClassVar[_decorators.DecoratorInfos] = _decorators.DecoratorInfos()
    'Metadata containing the decorators defined on the model.\n    This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.'
    __pydantic_generic_metadata__: ClassVar[_generics.PydanticGenericMetadata]
    'Metadata for generic models; contains data used for a similar purpose to\n    __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.'
    __pydantic_parent_namespace__: ClassVar[Dict[str, Any] | None] = None
    'Parent namespace of the model, used for automatic rebuilding of models.'
    __pydantic_post_init__: ClassVar[None | Literal['model_post_init']]
    'The name of the post-init method for the model, if defined.'
    __pydantic_root_model__: ClassVar[bool] = False
    'Whether the model is a [`RootModel`][pydantic.root_model.RootModel].'
    __pydantic_serializer__: ClassVar[SchemaSerializer]
    'The `pydantic-core` `SchemaSerializer` used to dump instances of the model.'
    __pydantic_validator__: ClassVar[SchemaValidator | PluggableSchemaValidator]
    'The `pydantic-core` `SchemaValidator` used to validate instances of the model.'
    __pydantic_fields__: ClassVar[Dict[str, FieldInfo]]
    'A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.\n    This replaces `Model.__fields__` from Pydantic V1.\n    '
    __pydantic_setattr_handlers__: ClassVar[Dict[str, Callable[[BaseModel, str, Any], None]]]
    '`__setattr__` handlers. Memoizing the handlers leads to a dramatic performance improvement in `__setattr__`'
    __pydantic_computed_fields__: ClassVar[Dict[str, ComputedFieldInfo]]
    'A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.'
    __pydantic_extra__: Dict[str, Any] | None = _model_construction.NoInitField(init=False)
    "A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra] is set to `'allow'`."
    __pydantic_fields_set__: set[str] = _model_construction.NoInitField(init=False)
    'The names of fields explicitly set during instantiation.'
    __pydantic_private__: Dict[str, Any] | None = _model_construction.NoInitField(init=False)
    'Values of private attributes set on the model instance.'
    if not TYPE_CHECKING:
        __pydantic_core_schema__ = _mock_val_ser.MockCoreSchema('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', code='base-model-instantiated')
        __pydantic_validator__ = _mock_val_ser.MockValSer('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', val_or_ser='validator', code='base-model-instantiated')
        __pydantic_serializer__ = _mock_val_ser.MockValSer('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', val_or_ser='serializer', code='base-model-instantiated')
    __slots__ = ('__dict__', '__pydantic_fields_set__', '__pydantic_extra__', '__pydantic_private__')

    def __init__(self, /, **data: Any) -> None:
        """Create a new model by parsing and validating input data from keyword arguments.

        Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
        validated to form a valid model.

        `self` is explicitly positional-only to allow `self` as a field name.
        """
        __tracebackhide__ = True
        validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
        if self is not validated_self:
            warnings.warn("A custom validator is returning a value other than `self`.\nReturning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\nSee the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.", stacklevel=2)
    __init__.__pydantic_base_init__ = True
    if not TYPE_CHECKING:

        def __getattr__(self, item: str) -> Any:
            private_attributes = object.__getattribute__(self, '__private_attributes__')
            if item in private_attributes:
                attribute = private_attributes[item]
                if hasattr(attribute, '__get__'):
                    return attribute.__get__(self, type(self))
                try:
                    return self.__pydantic_private__[item]
                except KeyError as exc:
                    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}') from exc
            else:
                try:
                    pydantic_extra = object.__getattribute__(self, '__pydantic_extra__')
                except AttributeError:
                    pydantic_extra = None
                if pydantic_extra and item in pydantic_extra:
                    return pydantic_extra[item]
                elif hasattr(self.__class__, item):
                    return super().__getattribute__(item)
                else:
                    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')

        def __setattr__(self, name: str, value: Any) -> None:
            if (setattr_handler := self.__pydantic_setattr_handlers__.get(name)) is not None:
                setattr_handler(self, name, value)
            elif (setattr_handler := self._setattr_handler(name, value)) is not None:
                setattr_handler(self, name, value)
                self.__pydantic_setattr_handlers__[name] = setattr_handler

        def _setattr_handler(self, name: str, value: Any) -> Callable[[BaseModel, str, Any], None] | None:
            """Get a handler for setting an attribute on the model instance.

            Returns:
                A handler for setting an attribute on the model instance. Used for memoization of the handler.
                Memoizing the handlers leads to a dramatic performance improvement in `__setattr__`
                Returns `None` when memoization is not safe, then the attribute is set directly.
            """
            cls = self.__class__
            if name in cls.__class_vars__:
                raise AttributeError(f'{name!r} is a ClassVar of `{cls.__name__}` and cannot be set on an instance. If you want to set a value on the class, use `{cls.__name__}.{name} = value`.')
            elif not _fields.is_valid_field_name(name):
                if (attribute := cls.__private_attributes__.get(name)) is not None:
                    if hasattr(attribute, '__set__'):
                        return lambda model, _name, val: attribute.__set__(model, val)
                    else:
                        return _SIMPLE_SETATTR_HANDLERS['private']
                else:
                    _object_setattr(self, name, value)
                    return None
            attr = getattr(cls, name, None)
            if isinstance(attr, cached_property):
                return _SIMPLE_SETATTR_HANDLERS['cached_property']
            _check_frozen(cls, name, value)
            if isinstance(attr, property):
                return lambda model, _name, val: attr.__set__(model, val)
            elif cls.model_config.get('validate_assignment'):
                return _SIMPLE_SETATTR_HANDLERS['validate_assignment']
            elif name not in cls.__pydantic_fields__:
                if cls.model_config.get('extra') != 'allow':
                    raise ValueError(f'"{cls.__name__}" object has no field "{name}"')
                elif attr is None:
                    self.__pydantic_extra__[name] = value
                    return None
                else:
                    return _SIMPLE_SETATTR_HANDLERS['extra_known']
            else:
                return _SIMPLE_SETATTR_HANDLERS['model_field']

        def __delattr__(self, item: str) -> Any:
            cls = self.__class__
            if item in self.__private_attributes__:
                attribute = self.__private_attributes__[item]
                if hasattr(attribute, '__delete__'):
                    attribute.__delete__(self)
                    return
                try:
                    del self.__pydantic_private__[item]
                    return
                except KeyError as exc:
                    raise AttributeError(f'{cls.__name__!r} object has no attribute {item!r}') from exc
            attr = getattr(cls, item, None)
            if isinstance(attr, cached_property):
                return object.__delattr__(self, item)
            _check_frozen(cls, name=item, value=None)
            if item in self.__pydantic_fields__:
                object.__delattr__(self, item)
            elif self.__pydantic_extra__ is not None and item in self.__pydantic_extra__:
                del self.__pydantic_extra__[item]
            else:
                try:
                    object.__delattr__(self, item)
                except AttributeError:
                    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')

        def __replace__(self, **changes: Any) -> Self:
            return self.model_copy(update=changes)
    if not TYPE_CHECKING:

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, BaseModel):
                self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
                other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
                if not (self_type == other_type and getattr(self, '__pydantic_private__', None) == getattr(other, '__pydantic_private__', None) and (self.__pydantic_extra__ == other.__pydantic_extra__)):
                    return False
                if self.__dict__ == other.__dict__:
                    return True
                model_fields = type(self).__pydantic_fields__.keys()
                if self.__dict__.keys() <= model_fields and other.__dict__.keys() <= model_fields:
                    return False
                getter = operator.itemgetter(*model_fields) if model_fields else lambda _: _utils._SENTINEL
                try:
                    return getter(self.__dict__) == getter(other.__dict__)
                except KeyError:
                    self_fields_proxy = _utils.SafeGetItemProxy(self.__dict__)
                    other_fields_proxy = _utils.SafeGetItemProxy(other.__dict__)
                    return getter(self_fields_proxy) == getter(other_fields_proxy)
            else:
                return NotImplemented
    if TYPE_CHECKING:

        def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
            """This signature is included purely to help type-checkers check arguments to class declaration, which
            provides a way to conveniently set model_config key/value pairs.

            ```python
            from pydantic import BaseModel

            class MyModel(BaseModel, extra='allow'): ...
            ```

            However, this may be deceiving, since the _actual_ calls to `__init_subclass__` will not receive any
            of the config arguments, and will only receive any keyword arguments passed during class initialization
            that are _not_ expected keys in ConfigDict. (This is due to the way `ModelMetaclass.__new__` works.)

            Args:
                **kwargs: Keyword arguments passed to the class definition, which set model_config

            Note:
                You may want to override `__pydantic_init_subclass__` instead, which behaves similarly but is called
                *after* the class is fully initialized.
            """
    __repr_name__ = _repr.Representation.__repr_name__
    __repr_recursion__ = _repr.Representation.__repr_recursion__
    __repr_str__ = _repr.Representation.__repr_str__
    __pretty__ = _repr.Representation.__pretty__
    __rich_repr__ = _repr.Representation.__rich_repr__
