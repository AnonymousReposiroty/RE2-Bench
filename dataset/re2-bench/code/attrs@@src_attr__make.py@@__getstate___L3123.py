from ._compat import (
    PY_3_10_PLUS,
    PY_3_11_PLUS,
    PY_3_13_PLUS,
    _AnnotationExtractor,
    _get_annotations,
    get_generic_base,
)

class Converter:
    """
    Stores a converter callable.

    Allows for the wrapped converter to take additional arguments. The
    arguments are passed in the order they are documented.

    Args:
        converter (Callable): A callable that converts the passed value.

        takes_self (bool):
            Pass the partially initialized instance that is being initialized
            as a positional argument. (default: `False`)

        takes_field (bool):
            Pass the field definition (an :class:`Attribute`) into the
            converter as a positional argument. (default: `False`)

    .. versionadded:: 24.1.0
    """
    __slots__ = ('__call__', '_first_param_type', '_global_name', 'converter', 'takes_field', 'takes_self')

    def __init__(self, converter, *, takes_self=False, takes_field=False):
        self.converter = converter
        self.takes_self = takes_self
        self.takes_field = takes_field
        ex = _AnnotationExtractor(converter)
        self._first_param_type = ex.get_first_param_type()
        if not (self.takes_self or self.takes_field):
            self.__call__ = lambda value, _, __: self.converter(value)
        elif self.takes_self and (not self.takes_field):
            self.__call__ = lambda value, instance, __: self.converter(value, instance)
        elif not self.takes_self and self.takes_field:
            self.__call__ = lambda value, __, field: self.converter(value, field)
        else:
            self.__call__ = lambda value, instance, field: self.converter(value, instance, field)
        rt = ex.get_return_type()
        if rt is not None:
            self.__call__.__annotations__['return'] = rt

    def __getstate__(self):
        """
        Return a dict containing only converter and takes_self -- the rest gets
        computed when loading.
        """
        return {'converter': self.converter, 'takes_self': self.takes_self, 'takes_field': self.takes_field}
