import collections.abc as cabc
import inspect
import typing as t
from . import types
from ._utils import UNSET
from .termui import prompt

class Option(Parameter):
    """Options are usually optional values on the command line and
    have some extra features that arguments don't have.

    All other parameters are passed onwards to the parameter constructor.

    :param show_default: Show the default value for this option in its
        help text. Values are not shown by default, unless
        :attr:`Context.show_default` is ``True``. If this value is a
        string, it shows that string in parentheses instead of the
        actual value. This is particularly useful for dynamic options.
        For single option boolean flags, the default remains hidden if
        its value is ``False``.
    :param show_envvar: Controls if an environment variable should be
        shown on the help page and error messages.
        Normally, environment variables are not shown.
    :param prompt: If set to ``True`` or a non empty string then the
        user will be prompted for input. If set to ``True`` the prompt
        will be the option name capitalized. A deprecated option cannot be
        prompted.
    :param confirmation_prompt: Prompt a second time to confirm the
        value if it was prompted for. Can be set to a string instead of
        ``True`` to customize the message.
    :param prompt_required: If set to ``False``, the user will be
        prompted for input only when the option was specified as a flag
        without a value.
    :param hide_input: If this is ``True`` then the input on the prompt
        will be hidden from the user. This is useful for password input.
    :param is_flag: forces this option to act as a flag.  The default is
                    auto detection.
    :param flag_value: which value should be used for this flag if it's
                       enabled.  This is set to a boolean automatically if
                       the option string contains a slash to mark two options.
    :param multiple: if this is set to `True` then the argument is accepted
                     multiple times and recorded.  This is similar to ``nargs``
                     in how it works but supports arbitrary number of
                     arguments.
    :param count: this flag makes an option increment an integer.
    :param allow_from_autoenv: if this is enabled then the value of this
                               parameter will be pulled from an environment
                               variable in case a prefix is defined on the
                               context.
    :param help: the help string.
    :param hidden: hide this option from help outputs.
    :param attrs: Other command arguments described in :class:`Parameter`.

    .. versionchanged:: 8.2
        ``envvar`` used with ``flag_value`` will always use the ``flag_value``,
        previously it would use the value of the environment variable.

    .. versionchanged:: 8.1
        Help text indentation is cleaned here instead of only in the
        ``@option`` decorator.

    .. versionchanged:: 8.1
        The ``show_default`` parameter overrides
        ``Context.show_default``.

    .. versionchanged:: 8.1
        The default of a single option boolean flag is not shown if the
        default value is ``False``.

    .. versionchanged:: 8.0.1
        ``type`` is detected from ``flag_value`` if given.
    """
    param_type_name = 'option'

    def __init__(self, param_decls: cabc.Sequence[str] | None=None, show_default: bool | str | None=None, prompt: bool | str=False, confirmation_prompt: bool | str=False, prompt_required: bool=True, hide_input: bool=False, is_flag: bool | None=None, flag_value: t.Any=UNSET, multiple: bool=False, count: bool=False, allow_from_autoenv: bool=True, type: types.ParamType | t.Any | None=None, help: str | None=None, hidden: bool=False, show_choices: bool=True, show_envvar: bool=False, deprecated: bool | str=False, **attrs: t.Any) -> None:
        if help:
            help = inspect.cleandoc(help)
        super().__init__(param_decls, type=type, multiple=multiple, deprecated=deprecated, **attrs)
        if prompt is True:
            if self.name is None:
                raise TypeError("'name' is required with 'prompt=True'.")
            prompt_text: str | None = self.name.replace('_', ' ').capitalize()
        elif prompt is False:
            prompt_text = None
        else:
            prompt_text = prompt
        if deprecated:
            deprecated_message = f'(DEPRECATED: {deprecated})' if isinstance(deprecated, str) else '(DEPRECATED)'
            help = help + deprecated_message if help is not None else deprecated_message
        self.prompt = prompt_text
        self.confirmation_prompt = confirmation_prompt
        self.prompt_required = prompt_required
        self.hide_input = hide_input
        self.hidden = hidden
        self._flag_needs_value = self.prompt is not None and (not self.prompt_required)
        if is_flag is None:
            if flag_value is not UNSET:
                is_flag = True
            elif self._flag_needs_value:
                is_flag = False
            elif self.secondary_opts:
                is_flag = True
        elif is_flag is False and (not self._flag_needs_value):
            self._flag_needs_value = self.default is UNSET
        if is_flag:
            if self.default is UNSET and (not self.required) and (not self.prompt):
                if multiple:
                    self.default = ()
            if type is None:
                if flag_value is UNSET:
                    self.type: types.ParamType = types.BoolParamType()
                elif isinstance(flag_value, bool):
                    self.type = types.BoolParamType()
                else:
                    self.type = types.convert_type(None, flag_value)
        self.is_flag: bool = bool(is_flag)
        self.is_bool_flag: bool = bool(is_flag and isinstance(self.type, types.BoolParamType))
        self.flag_value: t.Any = flag_value
        if self.is_bool_flag:
            if self.default is UNSET and (not self.required):
                self.default = False
        if self.default is True and self.flag_value is not UNSET:
            self.default = self.flag_value
        if self.flag_value is UNSET:
            if self.is_flag:
                self.flag_value = True
            else:
                self.flag_value = None
        self.count = count
        if count:
            if type is None:
                self.type = types.IntRange(min=0)
            if self.default is UNSET:
                self.default = 0
        self.allow_from_autoenv = allow_from_autoenv
        self.help = help
        self.show_default = show_default
        self.show_choices = show_choices
        self.show_envvar = show_envvar
        if __debug__:
            if deprecated and prompt:
                raise ValueError('`deprecated` options cannot use `prompt`.')
            if self.nargs == -1:
                raise TypeError('nargs=-1 is not supported for options.')
            if not self.is_bool_flag and self.secondary_opts:
                raise TypeError('Secondary flag is not valid for non-boolean flag.')
            if self.is_bool_flag and self.hide_input and (self.prompt is not None):
                raise TypeError("'prompt' with 'hide_input' is not valid for boolean flag.")
            if self.count:
                if self.multiple:
                    raise TypeError("'count' is not valid with 'multiple'.")
                if self.is_flag:
                    raise TypeError("'count' is not valid with 'is_flag'.")

    def to_info_dict(self) -> dict[str, t.Any]:
        """
        .. versionchanged:: 8.3.0
            Returns ``None`` for the :attr:`flag_value` if it was not set.
        """
        info_dict = super().to_info_dict()
        info_dict.update(help=self.help, prompt=self.prompt, is_flag=self.is_flag, flag_value=self.flag_value if self.flag_value is not UNSET else None, count=self.count, hidden=self.hidden)
        return info_dict
