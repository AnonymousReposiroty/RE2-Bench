import datetime
import re
from collections import deque
from collections.abc import Container
from dataclasses import dataclass
from functools import cached_property, partial
from re import Pattern
from typing import TYPE_CHECKING, Annotated, Any, Callable, Generic, Protocol, TypeVar, Union, overload
import annotated_types
from pydantic_core import core_schema as cs
from pydantic._internal._internal_dataclass import slots_true as _slots_true
from pydantic import GetCoreSchemaHandler
from types import EllipsisType

@dataclass(**_slots_true)
class _Pipeline(Generic[_InT, _OutT]):
    """Abstract representation of a chain of validation, transformation, and parsing steps."""

    _steps: tuple[_Step, ...]

    def transform(
        self,
        func: Callable[[_OutT], _NewOutT],
    ) -> _Pipeline[_InT, _NewOutT]:
        """Transform the output of the previous step.

        If used as the first step in a pipeline, the type of the field is used.
        That is, the transformation is applied to after the value is parsed to the field's type.
        """
        return _Pipeline[_InT, _NewOutT](self._steps + (_Transform(func),))

    @overload
    def validate_as(self, tp: type[_NewOutT], *, strict: bool = ...) -> _Pipeline[_InT, _NewOutT]: ...

    @overload
    def validate_as(self, tp: EllipsisType, *, strict: bool = ...) -> _Pipeline[_InT, Any]:  # type: ignore
        ...

    def validate_as(self, tp: type[_NewOutT] | EllipsisType, *, strict: bool = False) -> _Pipeline[_InT, Any]:  # type: ignore
        """Validate / parse the input into a new type.

        If no type is provided, the type of the field is used.

        Types are parsed in Pydantic's `lax` mode by default,
        but you can enable `strict` mode by passing `strict=True`.
        """
        if isinstance(tp, EllipsisType):
            return _Pipeline[_InT, Any](self._steps + (_ValidateAs(_FieldTypeMarker, strict=strict),))
        return _Pipeline[_InT, _NewOutT](self._steps + (_ValidateAs(tp, strict=strict),))

    def validate_as_deferred(self, func: Callable[[], type[_NewOutT]]) -> _Pipeline[_InT, _NewOutT]:
        """Parse the input into a new type, deferring resolution of the type until the current class
        is fully defined.

        This is useful when you need to reference the class in it's own type annotations.
        """
        return _Pipeline[_InT, _NewOutT](self._steps + (_ValidateAsDefer(func),))

    # constraints
    @overload
    def constrain(self: _Pipeline[_InT, _NewOutGe], constraint: annotated_types.Ge) -> _Pipeline[_InT, _NewOutGe]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutGt], constraint: annotated_types.Gt) -> _Pipeline[_InT, _NewOutGt]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutLe], constraint: annotated_types.Le) -> _Pipeline[_InT, _NewOutLe]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutLt], constraint: annotated_types.Lt) -> _Pipeline[_InT, _NewOutLt]: ...

    @overload
    def constrain(
        self: _Pipeline[_InT, _NewOutLen], constraint: annotated_types.Len
    ) -> _Pipeline[_InT, _NewOutLen]: ...

    @overload
    def constrain(
        self: _Pipeline[_InT, _NewOutT], constraint: annotated_types.MultipleOf
    ) -> _Pipeline[_InT, _NewOutT]: ...

    @overload
    def constrain(
        self: _Pipeline[_InT, _NewOutDatetime], constraint: annotated_types.Timezone
    ) -> _Pipeline[_InT, _NewOutDatetime]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: annotated_types.Predicate) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(
        self: _Pipeline[_InT, _NewOutInterval], constraint: annotated_types.Interval
    ) -> _Pipeline[_InT, _NewOutInterval]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _Eq) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _NotEq) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _In) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _NotIn) -> _Pipeline[_InT, _OutT]: ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutT], constraint: Pattern[str]) -> _Pipeline[_InT, _NewOutT]: ...

    def constrain(self, constraint: _ConstraintAnnotation) -> Any:
        """Constrain a value to meet a certain condition.

        We support most conditions from `annotated_types`, as well as regular expressions.

        Most of the time you'll be calling a shortcut method like `gt`, `lt`, `len`, etc
        so you don't need to call this directly.
        """
        return _Pipeline[_InT, _OutT](self._steps + (_Constraint(constraint),))

    def predicate(self: _Pipeline[_InT, _NewOutT], func: Callable[[_NewOutT], bool]) -> _Pipeline[_InT, _NewOutT]:
        """Constrain a value to meet a certain predicate."""
        return self.constrain(annotated_types.Predicate(func))

    def gt(self: _Pipeline[_InT, _NewOutGt], gt: _NewOutGt) -> _Pipeline[_InT, _NewOutGt]:
        """Constrain a value to be greater than a certain value."""
        return self.constrain(annotated_types.Gt(gt))

    def lt(self: _Pipeline[_InT, _NewOutLt], lt: _NewOutLt) -> _Pipeline[_InT, _NewOutLt]:
        """Constrain a value to be less than a certain value."""
        return self.constrain(annotated_types.Lt(lt))

    def ge(self: _Pipeline[_InT, _NewOutGe], ge: _NewOutGe) -> _Pipeline[_InT, _NewOutGe]:
        """Constrain a value to be greater than or equal to a certain value."""
        return self.constrain(annotated_types.Ge(ge))

    def le(self: _Pipeline[_InT, _NewOutLe], le: _NewOutLe) -> _Pipeline[_InT, _NewOutLe]:
        """Constrain a value to be less than or equal to a certain value."""
        return self.constrain(annotated_types.Le(le))

    def len(self: _Pipeline[_InT, _NewOutLen], min_len: int, max_len: int | None = None) -> _Pipeline[_InT, _NewOutLen]:
        """Constrain a value to have a certain length."""
        return self.constrain(annotated_types.Len(min_len, max_len))

    @overload
    def multiple_of(self: _Pipeline[_InT, _NewOutDiv], multiple_of: _NewOutDiv) -> _Pipeline[_InT, _NewOutDiv]: ...

    @overload
    def multiple_of(self: _Pipeline[_InT, _NewOutMod], multiple_of: _NewOutMod) -> _Pipeline[_InT, _NewOutMod]: ...

    def multiple_of(self: _Pipeline[_InT, Any], multiple_of: Any) -> _Pipeline[_InT, Any]:
        """Constrain a value to be a multiple of a certain number."""
        return self.constrain(annotated_types.MultipleOf(multiple_of))

    def eq(self: _Pipeline[_InT, _OutT], value: _OutT) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to be equal to a certain value."""
        return self.constrain(_Eq(value))

    def not_eq(self: _Pipeline[_InT, _OutT], value: _OutT) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to not be equal to a certain value."""
        return self.constrain(_NotEq(value))

    def in_(self: _Pipeline[_InT, _OutT], values: Container[_OutT]) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to be in a certain set."""
        return self.constrain(_In(values))

    def not_in(self: _Pipeline[_InT, _OutT], values: Container[_OutT]) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to not be in a certain set."""
        return self.constrain(_NotIn(values))

    # timezone methods
    def datetime_tz_naive(self: _Pipeline[_InT, datetime.datetime]) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(None))

    def datetime_tz_aware(self: _Pipeline[_InT, datetime.datetime]) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(...))

    def datetime_tz(
        self: _Pipeline[_InT, datetime.datetime], tz: datetime.tzinfo
    ) -> _Pipeline[_InT, datetime.datetime]:
        return self.constrain(annotated_types.Timezone(tz))  # type: ignore

    def datetime_with_tz(
        self: _Pipeline[_InT, datetime.datetime], tz: datetime.tzinfo | None
    ) -> _Pipeline[_InT, datetime.datetime]:
        return self.transform(partial(datetime.datetime.replace, tzinfo=tz))

    # string methods
    def str_lower(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
        return self.transform(str.lower)

    def str_upper(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
        return self.transform(str.upper)

    def str_title(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
        return self.transform(str.title)

    def str_strip(self: _Pipeline[_InT, str]) -> _Pipeline[_InT, str]:
        return self.transform(str.strip)

    def str_pattern(self: _Pipeline[_InT, str], pattern: str) -> _Pipeline[_InT, str]:
        return self.constrain(re.compile(pattern))

    def str_contains(self: _Pipeline[_InT, str], substring: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: substring in v)

    def str_starts_with(self: _Pipeline[_InT, str], prefix: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: v.startswith(prefix))

    def str_ends_with(self: _Pipeline[_InT, str], suffix: str) -> _Pipeline[_InT, str]:
        return self.predicate(lambda v: v.endswith(suffix))

    # operators
    def otherwise(self, other: _Pipeline[_OtherIn, _OtherOut]) -> _Pipeline[_InT | _OtherIn, _OutT | _OtherOut]:
        """Combine two validation chains, returning the result of the first chain if it succeeds, and the second chain if it fails."""
        return _Pipeline((_PipelineOr(self, other),))

    __or__ = otherwise

    def then(self, other: _Pipeline[_OutT, _OtherOut]) -> _Pipeline[_InT, _OtherOut]:
        """Pipe the result of one validation chain into another."""
        return _Pipeline((_PipelineAnd(self, other),))

    __and__ = then

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> cs.CoreSchema:
        queue = deque(self._steps)

        s = None

        while queue:
            step = queue.popleft()
            s = _apply_step(step, s, handler, source_type)

        s = s or cs.any_schema()
        return s

    def __supports_type__(self, _: _OutT) -> bool:
        raise NotImplementedError
