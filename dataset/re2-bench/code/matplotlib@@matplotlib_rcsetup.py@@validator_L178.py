from matplotlib import _api, cbook

def _make_type_validator(cls, *, allow_none=False):
    """
    Return a validator that converts inputs to *cls* or raises (and possibly
    allows ``None`` as well).
    """

    def validator(s):
        if (allow_none and
                (s is None or cbook._str_lower_equal(s, "none"))):
            if cbook._str_lower_equal(s, "none") and s != "None":
                _api.warn_deprecated(
                    "3.11",
                    message=f"Using the capitalization {s!r} in matplotlibrc for "
                            "*None* is deprecated in %(removal)s and will lead to an "
                            "error from version 3.13 onward. Please use 'None' "
                            "instead."
                )
            return None
        if cls is str and not isinstance(s, str):
            raise ValueError(f'Could not convert {s!r} to str')
        try:
            return cls(s)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f'Could not convert {s!r} to {cls.__name__}') from e

    validator.__name__ = f"validate_{cls.__name__}"
    if allow_none:
        validator.__name__ += "_or_None"
    validator.__qualname__ = (
        validator.__qualname__.rsplit(".", 1)[0] + "." + validator.__name__)
    return validator
