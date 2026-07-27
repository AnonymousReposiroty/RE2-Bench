class Options(_Constraint):
    """Constraint representing a finite set of instances of a given type.

    Parameters
    ----------
    type : type

    options : set
        The set of valid scalars.

    deprecated : set or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, type, options, *, deprecated=None):
        super().__init__()
        self.type = type
        self.options = options
        self.deprecated = deprecated or set()
        if self.deprecated - self.options:
            raise ValueError('The deprecated options must be a subset of the options.')

    def _mark_if_deprecated(self, option):
        """Add a deprecated mark to an option if needed."""
        option_str = f'{option!r}'
        if option in self.deprecated:
            option_str = f'{option_str} (deprecated)'
        return option_str
