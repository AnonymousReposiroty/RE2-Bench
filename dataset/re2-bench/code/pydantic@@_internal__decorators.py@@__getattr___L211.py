from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Literal, TypeVar, Union
from ..fields import ComputedFieldInfo
from ..fields import ComputedFieldInfo
from ..fields import ComputedFieldInfo

@dataclass  # can't use slots here since we set attributes on `__post_init__`
class PydanticDescriptorProxy(Generic[ReturnType]):
    """Wrap a classmethod, staticmethod, property or unbound function
    and act as a descriptor that allows us to detect decorated items
    from the class' attributes.

    This class' __get__ returns the wrapped item's __get__ result,
    which makes it transparent for classmethods and staticmethods.

    Attributes:
        wrapped: The decorator that has to be wrapped.
        decorator_info: The decorator info.
        shim: A wrapper function to wrap V1 style function.
    """

    wrapped: DecoratedType[ReturnType]
    decorator_info: DecoratorInfo
    shim: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None

    def __post_init__(self):
        for attr in 'setter', 'deleter':
            if hasattr(self.wrapped, attr):
                f = partial(self._call_wrapped_attr, name=attr)
                setattr(self, attr, f)

    def _call_wrapped_attr(self, func: Callable[[Any], None], *, name: str) -> PydanticDescriptorProxy[ReturnType]:
        self.wrapped = getattr(self.wrapped, name)(func)
        if isinstance(self.wrapped, property):
            # update ComputedFieldInfo.wrapped_property
            from ..fields import ComputedFieldInfo

            if isinstance(self.decorator_info, ComputedFieldInfo):
                self.decorator_info.wrapped_property = self.wrapped
        return self

    def __get__(self, obj: object | None, obj_type: type[object] | None = None) -> PydanticDescriptorProxy[ReturnType]:
        try:
            return self.wrapped.__get__(obj, obj_type)  # pyright: ignore[reportReturnType]
        except AttributeError:
            # not a descriptor, e.g. a partial object
            return self.wrapped  # type: ignore[return-value]

    def __set_name__(self, instance: Any, name: str) -> None:
        if hasattr(self.wrapped, '__set_name__'):
            self.wrapped.__set_name__(instance, name)  # pyright: ignore[reportFunctionMemberAccess]

    def __getattr__(self, name: str, /) -> Any:
        """Forward checks for __isabstractmethod__ and such."""
        return getattr(self.wrapped, name)
