import shlex

def split_arg_string(string: str) -> list[str]:
    """Split an argument string as with :func:`shlex.split`, but don't
    fail if the string is incomplete. Ignores a missing closing quote or
    incomplete escape sequence and uses the partial token as-is.

    .. code-block:: python

        split_arg_string("example 'my file")
        ["example", "my file"]

        split_arg_string("example my\\")
        ["example", "my"]

    :param string: String to split.

    .. versionchanged:: 8.2
        Moved to ``shell_completion`` from ``parser``.
    """
    import shlex

    lex = shlex.shlex(string, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    out = []

    try:
        for token in lex:
            out.append(token)
    except ValueError:
        # Raised when end-of-string is reached in an invalid state. Use
        # the partial token as-is. The quote or escape character is in
        # lex.state, not lex.token.
        out.append(lex.token)

    return out
