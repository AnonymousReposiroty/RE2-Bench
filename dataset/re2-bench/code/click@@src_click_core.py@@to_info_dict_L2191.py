import collections.abc as cabc
import typing as t
from . import types
from ._utils import UNSET
from .shell_completion import CompletionItem
from click.shell_completion import CompletionItem
from .shell_completion import shell_complete
from click.shell_completion import CompletionItem
from click.shell_completion import CompletionItem

class Parameter:
    """A parameter to a command comes in two versions: they are either
    :class:`Option`\\s or :class:`Argument`\\s.  Other subclasses are currently
    not supported by design as some of the internals for parsing are
    intentionally not finalized.

    Some settings are supported by both options and arguments.

    :param param_decls: the parameter declarations for this option or
                        argument.  This is a list of flags or argument
                        names.
    :param type: the type that should be used.  Either a :class:`ParamType`
                 or a Python type.  The latter is converted into the former
                 automatically if supported.
    :param required: controls if this is optional or not.
    :param default: the default value if omitted.  This can also be a callable,
                    in which case it's invoked when the default is needed
                    without any arguments.
    :param callback: A function to further process or validate the value
        after type conversion. It is called as ``f(ctx, param, value)``
        and must return the value. It is called for all sources,
        including prompts.
    :param nargs: the number of arguments to match.  If not ``1`` the return
                  value is a tuple instead of single value.  The default for
                  nargs is ``1`` (except if the type is a tuple, then it's
                  the arity of the tuple). If ``nargs=-1``, all remaining
                  parameters are collected.
    :param metavar: how the value is represented in the help page.
    :param expose_value: if this is `True` then the value is passed onwards
                         to the command callback and stored on the context,
                         otherwise it's skipped.
    :param is_eager: eager values are processed before non eager ones.  This
                     should not be set for arguments or it will inverse the
                     order of processing.
    :param envvar: environment variable(s) that are used to provide a default value for
        this parameter. This can be a string or a sequence of strings. If a sequence is
        given, only the first non-empty environment variable is used for the parameter.
    :param shell_complete: A function that returns custom shell
        completions. Used instead of the param's type completion if
        given. Takes ``ctx, param, incomplete`` and must return a list
        of :class:`~click.shell_completion.CompletionItem` or a list of
        strings.
    :param deprecated: If ``True`` or non-empty string, issues a message
                        indicating that the argument is deprecated and highlights
                        its deprecation in --help. The message can be customized
                        by using a string as the value. A deprecated parameter
                        cannot be required, a ValueError will be raised otherwise.

    .. versionchanged:: 8.2.0
        Introduction of ``deprecated``.

    .. versionchanged:: 8.2
        Adding duplicate parameter names to a :class:`~click.core.Command` will
        result in a ``UserWarning`` being shown.

    .. versionchanged:: 8.2
        Adding duplicate parameter names to a :class:`~click.core.Command` will
        result in a ``UserWarning`` being shown.

    .. versionchanged:: 8.0
        ``process_value`` validates required parameters and bounded
        ``nargs``, and invokes the parameter callback before returning
        the value. This allows the callback to validate prompts.
        ``full_process_value`` is removed.

    .. versionchanged:: 8.0
        ``autocompletion`` is renamed to ``shell_complete`` and has new
        semantics described above. The old name is deprecated and will
        be removed in 8.1, until then it will be wrapped to match the
        new requirements.

    .. versionchanged:: 8.0
        For ``multiple=True, nargs>1``, the default must be a list of
        tuples.

    .. versionchanged:: 8.0
        Setting a default is no longer required for ``nargs>1``, it will
        default to ``None``. ``multiple=True`` or ``nargs=-1`` will
        default to ``()``.

    .. versionchanged:: 7.1
        Empty environment variables are ignored rather than taking the
        empty string value. This makes it possible for scripts to clear
        variables if they can't unset them.

    .. versionchanged:: 2.0
        Changed signature for parameter callback to also be passed the
        parameter. The old callback format will still work, but it will
        raise a warning to give you a chance to migrate the code easier.
    """
    param_type_name = 'parameter'

    def __init__(self, param_decls: cabc.Sequence[str] | None=None, type: types.ParamType | t.Any | None=None, required: bool=False, default: t.Any | t.Callable[[], t.Any] | None=UNSET, callback: t.Callable[[Context, Parameter, t.Any], t.Any] | None=None, nargs: int | None=None, multiple: bool=False, metavar: str | None=None, expose_value: bool=True, is_eager: bool=False, envvar: str | cabc.Sequence[str] | None=None, shell_complete: t.Callable[[Context, Parameter, str], list[CompletionItem] | list[str]] | None=None, deprecated: bool | str=False) -> None:
        self.name: str | None
        self.opts: list[str]
        self.secondary_opts: list[str]
        self.name, self.opts, self.secondary_opts = self._parse_decls(param_decls or (), expose_value)
        self.type: types.ParamType = types.convert_type(type, default)
        if nargs is None:
            if self.type.is_composite:
                nargs = self.type.arity
            else:
                nargs = 1
        self.required = required
        self.callback = callback
        self.nargs = nargs
        self.multiple = multiple
        self.expose_value = expose_value
        self.default: t.Any | t.Callable[[], t.Any] | None = default
        self.is_eager = is_eager
        self.metavar = metavar
        self.envvar = envvar
        self._custom_shell_complete = shell_complete
        self.deprecated = deprecated
        if __debug__:
            if self.type.is_composite and nargs != self.type.arity:
                raise ValueError(f"'nargs' must be {self.type.arity} (or None) for type {self.type!r}, but it was {nargs}.")
            if required and deprecated:
                raise ValueError(f"The {self.param_type_name} '{self.human_readable_name}' is deprecated and still required. A deprecated {self.param_type_name} cannot be required.")

    def to_info_dict(self) -> dict[str, t.Any]:
        """Gather information that could be useful for a tool generating
        user-facing documentation.

        Use :meth:`click.Context.to_info_dict` to traverse the entire
        CLI structure.

        .. versionchanged:: 8.3.0
            Returns ``None`` for the :attr:`default` if it was not set.

        .. versionadded:: 8.0
        """
        return {'name': self.name, 'param_type_name': self.param_type_name, 'opts': self.opts, 'secondary_opts': self.secondary_opts, 'type': self.type.to_info_dict(), 'required': self.required, 'nargs': self.nargs, 'multiple': self.multiple, 'default': self.default if self.default is not UNSET else None, 'envvar': self.envvar}

    def _parse_decls(self, decls: cabc.Sequence[str], expose_value: bool) -> tuple[str | None, list[str], list[str]]:
        raise NotImplementedError()

    @property
    def human_readable_name(self) -> str:
        """Returns the human readable name of this parameter.  This is the
        same as the name for options, but the metavar for arguments.
        """
        return self.name
