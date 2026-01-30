import collections.abc as cabc
import enum
import inspect
import os
import typing as t
from gettext import gettext as _
from . import types
from ._utils import FLAG_NEEDS_VALUE
from ._utils import UNSET
from .formatting import join_options
from .parser import _OptionParser
from .parser import _split_opt
from .termui import confirm
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

    param_type_name = "option"

    def __init__(
        self,
        param_decls: cabc.Sequence[str] | None = None,
        show_default: bool | str | None = None,
        prompt: bool | str = False,
        confirmation_prompt: bool | str = False,
        prompt_required: bool = True,
        hide_input: bool = False,
        is_flag: bool | None = None,
        flag_value: t.Any = UNSET,
        multiple: bool = False,
        count: bool = False,
        allow_from_autoenv: bool = True,
        type: types.ParamType | t.Any | None = None,
        help: str | None = None,
        hidden: bool = False,
        show_choices: bool = True,
        show_envvar: bool = False,
        deprecated: bool | str = False,
        **attrs: t.Any,
    ) -> None:
        if help:
            help = inspect.cleandoc(help)

        super().__init__(
            param_decls, type=type, multiple=multiple, deprecated=deprecated, **attrs
        )

        if prompt is True:
            if self.name is None:
                raise TypeError("'name' is required with 'prompt=True'.")

            prompt_text: str | None = self.name.replace("_", " ").capitalize()
        elif prompt is False:
            prompt_text = None
        else:
            prompt_text = prompt

        if deprecated:
            deprecated_message = (
                f"(DEPRECATED: {deprecated})"
                if isinstance(deprecated, str)
                else "(DEPRECATED)"
            )
            help = help + deprecated_message if help is not None else deprecated_message

        self.prompt = prompt_text
        self.confirmation_prompt = confirmation_prompt
        self.prompt_required = prompt_required
        self.hide_input = hide_input
        self.hidden = hidden

        # The _flag_needs_value property tells the parser that this option is a flag
        # that cannot be used standalone and needs a value. With this information, the
        # parser can determine whether to consider the next user-provided argument in
        # the CLI as a value for this flag or as a new option.
        # If prompt is enabled but not required, then it opens the possibility for the
        # option to gets its value from the user.
        self._flag_needs_value = self.prompt is not None and not self.prompt_required

        # Auto-detect if this is a flag or not.
        if is_flag is None:
            # Implicitly a flag because flag_value was set.
            if flag_value is not UNSET:
                is_flag = True
            # Not a flag, but when used as a flag it shows a prompt.
            elif self._flag_needs_value:
                is_flag = False
            # Implicitly a flag because secondary options names were given.
            elif self.secondary_opts:
                is_flag = True
        # The option is explicitly not a flag. But we do not know yet if it needs a
        # value or not. So we look at the default value to determine it.
        elif is_flag is False and not self._flag_needs_value:
            self._flag_needs_value = self.default is UNSET

        if is_flag:
            # Set missing default for flags if not explicitly required or prompted.
            if self.default is UNSET and not self.required and not self.prompt:
                if multiple:
                    self.default = ()

            # Auto-detect the type of the flag based on the flag_value.
            if type is None:
                # A flag without a flag_value is a boolean flag.
                if flag_value is UNSET:
                    self.type: types.ParamType = types.BoolParamType()
                # If the flag value is a boolean, use BoolParamType.
                elif isinstance(flag_value, bool):
                    self.type = types.BoolParamType()
                # Otherwise, guess the type from the flag value.
                else:
                    self.type = types.convert_type(None, flag_value)

        self.is_flag: bool = bool(is_flag)
        self.is_bool_flag: bool = bool(
            is_flag and isinstance(self.type, types.BoolParamType)
        )
        self.flag_value: t.Any = flag_value

        # Set boolean flag default to False if unset and not required.
        if self.is_bool_flag:
            if self.default is UNSET and not self.required:
                self.default = False

        # Support the special case of aligning the default value with the flag_value
        # for flags whose default is explicitly set to True. Note that as long as we
        # have this condition, there is no way a flag can have a default set to True,
        # and a flag_value set to something else. Refs:
        # https://github.com/pallets/click/issues/3024#issuecomment-3146199461
        # https://github.com/pallets/click/pull/3030/commits/06847da
        if self.default is True and self.flag_value is not UNSET:
            self.default = self.flag_value

        # Set the default flag_value if it is not set.
        if self.flag_value is UNSET:
            if self.is_flag:
                self.flag_value = True
            else:
                self.flag_value = None

        # Counting.
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
                raise ValueError("`deprecated` options cannot use `prompt`.")

            if self.nargs == -1:
                raise TypeError("nargs=-1 is not supported for options.")

            if not self.is_bool_flag and self.secondary_opts:
                raise TypeError("Secondary flag is not valid for non-boolean flag.")

            if self.is_bool_flag and self.hide_input and self.prompt is not None:
                raise TypeError(
                    "'prompt' with 'hide_input' is not valid for boolean flag."
                )

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
        info_dict.update(
            help=self.help,
            prompt=self.prompt,
            is_flag=self.is_flag,
            # We explicitly hide the :attr:`UNSET` value to the user, as we choose to
            # make it an implementation detail. And because ``to_info_dict`` has been
            # designed for documentation purposes, we return ``None`` instead.
            flag_value=self.flag_value if self.flag_value is not UNSET else None,
            count=self.count,
            hidden=self.hidden,
        )
        return info_dict

    def get_error_hint(self, ctx: Context) -> str:
        result = super().get_error_hint(ctx)
        if self.show_envvar and self.envvar is not None:
            result += f" (env var: '{self.envvar}')"
        return result

    def _parse_decls(
        self, decls: cabc.Sequence[str], expose_value: bool
    ) -> tuple[str | None, list[str], list[str]]:
        opts = []
        secondary_opts = []
        name = None
        possible_names = []

        for decl in decls:
            if decl.isidentifier():
                if name is not None:
                    raise TypeError(f"Name '{name}' defined twice")
                name = decl
            else:
                split_char = ";" if decl[:1] == "/" else "/"
                if split_char in decl:
                    first, second = decl.split(split_char, 1)
                    first = first.rstrip()
                    if first:
                        possible_names.append(_split_opt(first))
                        opts.append(first)
                    second = second.lstrip()
                    if second:
                        secondary_opts.append(second.lstrip())
                    if first == second:
                        raise ValueError(
                            f"Boolean option {decl!r} cannot use the"
                            " same flag for true/false."
                        )
                else:
                    possible_names.append(_split_opt(decl))
                    opts.append(decl)

        if name is None and possible_names:
            possible_names.sort(key=lambda x: -len(x[0]))  # group long options first
            name = possible_names[0][1].replace("-", "_").lower()
            if not name.isidentifier():
                name = None

        if name is None:
            if not expose_value:
                return None, opts, secondary_opts
            raise TypeError(
                f"Could not determine name for option with declarations {decls!r}"
            )

        if not opts and not secondary_opts:
            raise TypeError(
                f"No options defined but a name was passed ({name})."
                " Did you mean to declare an argument instead? Did"
                f" you mean to pass '--{name}'?"
            )

        return name, opts, secondary_opts

    def add_to_parser(self, parser: _OptionParser, ctx: Context) -> None:
        if self.multiple:
            action = "append"
        elif self.count:
            action = "count"
        else:
            action = "store"

        if self.is_flag:
            action = f"{action}_const"

            if self.is_bool_flag and self.secondary_opts:
                parser.add_option(
                    obj=self, opts=self.opts, dest=self.name, action=action, const=True
                )
                parser.add_option(
                    obj=self,
                    opts=self.secondary_opts,
                    dest=self.name,
                    action=action,
                    const=False,
                )
            else:
                parser.add_option(
                    obj=self,
                    opts=self.opts,
                    dest=self.name,
                    action=action,
                    const=self.flag_value,
                )
        else:
            parser.add_option(
                obj=self,
                opts=self.opts,
                dest=self.name,
                action=action,
                nargs=self.nargs,
            )

    def get_help_record(self, ctx: Context) -> tuple[str, str] | None:
        if self.hidden:
            return None

        any_prefix_is_slash = False

        def _write_opts(opts: cabc.Sequence[str]) -> str:
            nonlocal any_prefix_is_slash

            rv, any_slashes = join_options(opts)

            if any_slashes:
                any_prefix_is_slash = True

            if not self.is_flag and not self.count:
                rv += f" {self.make_metavar(ctx=ctx)}"

            return rv

        rv = [_write_opts(self.opts)]

        if self.secondary_opts:
            rv.append(_write_opts(self.secondary_opts))

        help = self.help or ""

        extra = self.get_help_extra(ctx)
        extra_items = []
        if "envvars" in extra:
            extra_items.append(
                _("env var: {var}").format(var=", ".join(extra["envvars"]))
            )
        if "default" in extra:
            extra_items.append(_("default: {default}").format(default=extra["default"]))
        if "range" in extra:
            extra_items.append(extra["range"])
        if "required" in extra:
            extra_items.append(_(extra["required"]))

        if extra_items:
            extra_str = "; ".join(extra_items)
            help = f"{help}  [{extra_str}]" if help else f"[{extra_str}]"

        return ("; " if any_prefix_is_slash else " / ").join(rv), help

    def get_help_extra(self, ctx: Context) -> types.OptionHelpExtra:
        extra: types.OptionHelpExtra = {}

        if self.show_envvar:
            envvar = self.envvar

            if envvar is None:
                if (
                    self.allow_from_autoenv
                    and ctx.auto_envvar_prefix is not None
                    and self.name is not None
                ):
                    envvar = f"{ctx.auto_envvar_prefix}_{self.name.upper()}"

            if envvar is not None:
                if isinstance(envvar, str):
                    extra["envvars"] = (envvar,)
                else:
                    extra["envvars"] = tuple(str(d) for d in envvar)

        # Temporarily enable resilient parsing to avoid type casting
        # failing for the default. Might be possible to extend this to
        # help formatting in general.
        resilient = ctx.resilient_parsing
        ctx.resilient_parsing = True

        try:
            default_value = self.get_default(ctx, call=False)
        finally:
            ctx.resilient_parsing = resilient

        show_default = False
        show_default_is_str = False

        if self.show_default is not None:
            if isinstance(self.show_default, str):
                show_default_is_str = show_default = True
            else:
                show_default = self.show_default
        elif ctx.show_default is not None:
            show_default = ctx.show_default

        if show_default_is_str or (
            show_default and (default_value not in (None, UNSET))
        ):
            if show_default_is_str:
                default_string = f"({self.show_default})"
            elif isinstance(default_value, (list, tuple)):
                default_string = ", ".join(str(d) for d in default_value)
            elif isinstance(default_value, enum.Enum):
                default_string = default_value.name
            elif inspect.isfunction(default_value):
                default_string = _("(dynamic)")
            elif self.is_bool_flag and self.secondary_opts:
                # For boolean flags that have distinct True/False opts,
                # use the opt without prefix instead of the value.
                default_string = _split_opt(
                    (self.opts if default_value else self.secondary_opts)[0]
                )[1]
            elif self.is_bool_flag and not self.secondary_opts and not default_value:
                default_string = ""
            elif default_value == "":
                default_string = '""'
            else:
                default_string = str(default_value)

            if default_string:
                extra["default"] = default_string

        if (
            isinstance(self.type, types._NumberRangeBase)
            # skip count with default range type
            and not (self.count and self.type.min == 0 and self.type.max is None)
        ):
            range_str = self.type._describe_range()

            if range_str:
                extra["range"] = range_str

        if self.required:
            extra["required"] = "required"

        return extra

    def prompt_for_value(self, ctx: Context) -> t.Any:
        """This is an alternative flow that can be activated in the full
        value processing if a value does not exist.  It will prompt the
        user until a valid value exists and then returns the processed
        value as result.
        """
        assert self.prompt is not None

        # Calculate the default before prompting anything to lock in the value before
        # attempting any user interaction.
        default = self.get_default(ctx)

        # A boolean flag can use a simplified [y/n] confirmation prompt.
        if self.is_bool_flag:
            # If we have no boolean default, we force the user to explicitly provide
            # one.
            if default in (UNSET, None):
                default = None
            # Nothing prevent you to declare an option that is simultaneously:
            # 1) auto-detected as a boolean flag,
            # 2) allowed to prompt, and
            # 3) still declare a non-boolean default.
            # This forced casting into a boolean is necessary to align any non-boolean
            # default to the prompt, which is going to be a [y/n]-style confirmation
            # because the option is still a boolean flag. That way, instead of [y/n],
            # we get [Y/n] or [y/N] depending on the truthy value of the default.
            # Refs: https://github.com/pallets/click/pull/3030#discussion_r2289180249
            else:
                default = bool(default)
            return confirm(self.prompt, default)

        # If show_default is set to True/False, provide this to `prompt` as well. For
        # non-bool values of `show_default`, we use `prompt`'s default behavior
        prompt_kwargs: t.Any = {}
        if isinstance(self.show_default, bool):
            prompt_kwargs["show_default"] = self.show_default

        return prompt(
            self.prompt,
            # Use ``None`` to inform the prompt() function to reiterate until a valid
            # value is provided by the user if we have no default.
            default=None if default is UNSET else default,
            type=self.type,
            hide_input=self.hide_input,
            show_choices=self.show_choices,
            confirmation_prompt=self.confirmation_prompt,
            value_proc=lambda x: self.process_value(ctx, x),
            **prompt_kwargs,
        )

    def resolve_envvar_value(self, ctx: Context) -> str | None:
        """:class:`Option` resolves its environment variable the same way as
        :func:`Parameter.resolve_envvar_value`, but it also supports
        :attr:`Context.auto_envvar_prefix`. If we could not find an environment from
        the :attr:`envvar` property, we fallback on :attr:`Context.auto_envvar_prefix`
        to build dynamiccaly the environment variable name using the
        :python:`{ctx.auto_envvar_prefix}_{self.name.upper()}` template.

        :meta private:
        """
        rv = super().resolve_envvar_value(ctx)

        if rv is not None:
            return rv

        if (
            self.allow_from_autoenv
            and ctx.auto_envvar_prefix is not None
            and self.name is not None
        ):
            envvar = f"{ctx.auto_envvar_prefix}_{self.name.upper()}"
            rv = os.environ.get(envvar)

            if rv:
                return rv

        return None

    def value_from_envvar(self, ctx: Context) -> t.Any:
        """For :class:`Option`, this method processes the raw environment variable
        string the same way as :func:`Parameter.value_from_envvar` does.

        But in the case of non-boolean flags, the value is analyzed to determine if the
        flag is activated or not, and returns a boolean of its activation, or the
        :attr:`flag_value` if the latter is set.

        This method also takes care of repeated options (i.e. options with
        :attr:`multiple` set to ``True``).

        :meta private:
        """
        rv = self.resolve_envvar_value(ctx)

        # Absent environment variable or an empty string is interpreted as unset.
        if rv is None:
            return None

        # Non-boolean flags are more liberal in what they accept. But a flag being a
        # flag, its envvar value still needs to be analyzed to determine if the flag is
        # activated or not.
        if self.is_flag and not self.is_bool_flag:
            # If the flag_value is set and match the envvar value, return it
            # directly.
            if self.flag_value is not UNSET and rv == self.flag_value:
                return self.flag_value
            # Analyze the envvar value as a boolean to know if the flag is
            # activated or not.
            return types.BoolParamType.str_to_bool(rv)

        # Split the envvar value if it is allowed to be repeated.
        value_depth = (self.nargs != 1) + bool(self.multiple)
        if value_depth > 0:
            multi_rv = self.type.split_envvar_value(rv)
            if self.multiple and self.nargs != 1:
                multi_rv = batch(multi_rv, self.nargs)  # type: ignore[assignment]

            return multi_rv

        return rv

    def consume_value(
        self, ctx: Context, opts: cabc.Mapping[str, Parameter]
    ) -> tuple[t.Any, ParameterSource]:
        """For :class:`Option`, the value can be collected from an interactive prompt
        if the option is a flag that needs a value (and the :attr:`prompt` property is
        set).

        Additionally, this method handles flag option that are activated without a
        value, in which case the :attr:`flag_value` is returned.

        :meta private:
        """
        value, source = super().consume_value(ctx, opts)

        # The parser will emit a sentinel value if the option is allowed to as a flag
        # without a value.
        if value is FLAG_NEEDS_VALUE:
            # If the option allows for a prompt, we start an interaction with the user.
            if self.prompt is not None and not ctx.resilient_parsing:
                value = self.prompt_for_value(ctx)
                source = ParameterSource.PROMPT
            # Else the flag takes its flag_value as value.
            else:
                value = self.flag_value
                source = ParameterSource.COMMANDLINE

        # A flag which is activated always returns the flag value, unless the value
        # comes from the explicitly sets default.
        elif (
            self.is_flag
            and value is True
            and not self.is_bool_flag
            and source not in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP)
        ):
            value = self.flag_value

        # Re-interpret a multiple option which has been sent as-is by the parser.
        # Here we replace each occurrence of value-less flags (marked by the
        # FLAG_NEEDS_VALUE sentinel) with the flag_value.
        elif (
            self.multiple
            and value is not UNSET
            and source not in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP)
            and any(v is FLAG_NEEDS_VALUE for v in value)
        ):
            value = [self.flag_value if v is FLAG_NEEDS_VALUE else v for v in value]
            source = ParameterSource.COMMANDLINE

        # The value wasn't set, or used the param's default, prompt for one to the user
        # if prompting is enabled.
        elif (
            (
                value is UNSET
                or source in (ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP)
            )
            and self.prompt is not None
            and (self.required or self.prompt_required)
            and not ctx.resilient_parsing
        ):
            value = self.prompt_for_value(ctx)
            source = ParameterSource.PROMPT

        return value, source

    def process_value(self, ctx: Context, value: t.Any) -> t.Any:
        # process_value has to be overridden on Options in order to capture
        # `value == UNSET` cases before `type_cast_value()` gets called.
        #
        # Refs:
        # https://github.com/pallets/click/issues/3069
        if self.is_flag and not self.required and self.is_bool_flag and value is UNSET:
            value = False

            if self.callback is not None:
                value = self.callback(ctx, self, value)

            return value

        # in the normal case, rely on Parameter.process_value
        return super().process_value(ctx, value)
