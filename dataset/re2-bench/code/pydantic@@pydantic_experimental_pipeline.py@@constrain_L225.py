from dataclasses import dataclass
from re import Pattern
from typing import TYPE_CHECKING, Annotated, Any, Callable, Generic, Protocol, TypeVar, Union, overload
import annotated_types
from pydantic._internal._internal_dataclass import slots_true as _slots_true

@dataclass(**_slots_true)
class _Pipeline(Generic[_InT, _OutT]):
    """Abstract representation of a chain of validation, transformation, and parsing steps."""
    _steps: tuple[_Step, ...]

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutGe], constraint: annotated_types.Ge) -> _Pipeline[_InT, _NewOutGe]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutGt], constraint: annotated_types.Gt) -> _Pipeline[_InT, _NewOutGt]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutLe], constraint: annotated_types.Le) -> _Pipeline[_InT, _NewOutLe]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutLt], constraint: annotated_types.Lt) -> _Pipeline[_InT, _NewOutLt]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutLen], constraint: annotated_types.Len) -> _Pipeline[_InT, _NewOutLen]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutT], constraint: annotated_types.MultipleOf) -> _Pipeline[_InT, _NewOutT]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutDatetime], constraint: annotated_types.Timezone) -> _Pipeline[_InT, _NewOutDatetime]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: annotated_types.Predicate) -> _Pipeline[_InT, _OutT]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutInterval], constraint: annotated_types.Interval) -> _Pipeline[_InT, _NewOutInterval]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _Eq) -> _Pipeline[_InT, _OutT]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _NotEq) -> _Pipeline[_InT, _OutT]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _In) -> _Pipeline[_InT, _OutT]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _OutT], constraint: _NotIn) -> _Pipeline[_InT, _OutT]:
        ...

    @overload
    def constrain(self: _Pipeline[_InT, _NewOutT], constraint: Pattern[str]) -> _Pipeline[_InT, _NewOutT]:
        ...

    def constrain(self, constraint: _ConstraintAnnotation) -> Any:
        """Constrain a value to meet a certain condition.

        We support most conditions from `annotated_types`, as well as regular expressions.

        Most of the time you'll be calling a shortcut method like `gt`, `lt`, `len`, etc
        so you don't need to call this directly.
        """
        return _Pipeline[_InT, _OutT](self._steps + (_Constraint(constraint),))
    __or__ = otherwise
    __and__ = then
