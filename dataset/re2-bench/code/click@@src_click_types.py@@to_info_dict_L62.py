import typing as t

class ParamType:
    """Represents the type of a parameter. Validates and converts values
    from the command line or Python into the correct type.

    To implement a custom type, subclass and implement at least the
    following:

    -   The :attr:`name` class attribute must be set.
    -   Calling an instance of the type with ``None`` must return
        ``None``. This is already implemented by default.
    -   :meth:`convert` must convert string values to the correct type.
    -   :meth:`convert` must accept values that are already the correct
        type.
    -   It must be able to convert a value if the ``ctx`` and ``param``
        arguments are ``None``. This can occur when converting prompt
        input.
    """
    is_composite: t.ClassVar[bool] = False
    arity: t.ClassVar[int] = 1
    name: str
    envvar_list_splitter: t.ClassVar[str | None] = None

    def to_info_dict(self) -> dict[str, t.Any]:
        """Gather information that could be useful for a tool generating
        user-facing documentation.

        Use :meth:`click.Context.to_info_dict` to traverse the entire
        CLI structure.

        .. versionadded:: 8.0
        """
        param_type = type(self).__name__.partition('ParamType')[0]
        param_type = param_type.partition('ParameterType')[0]
        if hasattr(self, 'name'):
            name = self.name
        else:
            name = param_type
        return {'param_type': param_type, 'name': name}
