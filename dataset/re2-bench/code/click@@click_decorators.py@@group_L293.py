import typing as t
from .core import Group

def group(
    name: str | _AnyCallable | None = None,
    cls: type[GrpType] | None = None,
    **attrs: t.Any,
) -> Group | t.Callable[[_AnyCallable], Group | GrpType]:
    """Creates a new :class:`Group` with a function as callback.  This
    works otherwise the same as :func:`command` just that the `cls`
    parameter is set to :class:`Group`.

    .. versionchanged:: 8.1
        This decorator can be applied without parentheses.
    """
    if cls is None:
        cls = t.cast("type[GrpType]", Group)

    if callable(name):
        return command(cls=cls, **attrs)(name)

    return command(name, cls, **attrs)
