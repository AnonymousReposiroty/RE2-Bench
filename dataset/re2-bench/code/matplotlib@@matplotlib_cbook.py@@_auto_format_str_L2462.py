def _auto_format_str(fmt, value):
    """
    Apply *value* to the format string *fmt*.

    This works both with unnamed %-style formatting and
    unnamed {}-style formatting. %-style formatting has priority.
    If *fmt* is %-style formattable that will be used. Otherwise,
    {}-formatting is applied. Strings without formatting placeholders
    are passed through as is.

    Examples
    --------
    >>> _auto_format_str('%.2f m', 0.2)
    '0.20 m'
    >>> _auto_format_str('{} m', 0.2)
    '0.2 m'
    >>> _auto_format_str('const', 0.2)
    'const'
    >>> _auto_format_str('%d or {}', 0.2)
    '0 or {}'
    """
    try:
        return fmt % (value,)
    except (TypeError, ValueError):
        return fmt.format(value)
