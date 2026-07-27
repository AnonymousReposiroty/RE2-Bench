from ._compat import (
    PY_3_10_PLUS,
    PY_3_11_PLUS,
    PY_3_13_PLUS,
    _AnnotationExtractor,
    _get_annotations,
    get_generic_base,
)
from .exceptions import (
    DefaultAlreadySetError,
    FrozenInstanceError,
    NotAnAttrsClassError,
    UnannotatedAttributeError,
)

def fields(cls):
    """
    Return the tuple of *attrs* attributes for a class.

    The tuple also allows accessing the fields by their names (see below for
    examples).

    Args:
        cls (type): Class to introspect.

    Raises:
        TypeError: If *cls* is not a class.

        attrs.exceptions.NotAnAttrsClassError:
            If *cls* is not an *attrs* class.

    Returns:
        tuple (with name accessors) of `attrs.Attribute`

    .. versionchanged:: 16.2.0 Returned tuple allows accessing the fields
       by name.
    .. versionchanged:: 23.1.0 Add support for generic classes.
    """
    generic_base = get_generic_base(cls)

    if generic_base is None and not isinstance(cls, type):
        msg = "Passed object must be a class."
        raise TypeError(msg)

    attrs = getattr(cls, "__attrs_attrs__", None)

    if attrs is None:
        if generic_base is not None:
            attrs = getattr(generic_base, "__attrs_attrs__", None)
            if attrs is not None:
                # Even though this is global state, stick it on here to speed
                # it up. We rely on `cls` being cached for this to be
                # efficient.
                cls.__attrs_attrs__ = attrs
                return attrs
        msg = f"{cls!r} is not an attrs-decorated class."
        raise NotAnAttrsClassError(msg)

    return attrs
