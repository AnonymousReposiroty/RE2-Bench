def strip_escape_sequences(text: str, /) -> str:
    r"""Remove the ANSI CSI colors and "erase in line" sequences.

    Other `escape sequences <https://en.wikipedia.org/wiki/ANSI_escape_code>`_
    (e.g., VT100-specific functions) are not supported. Only control sequences
    *natively* known to Sphinx (i.e., colour sequences used in Sphinx
    and "erase entire line" (``'\x1b[2K'``)) are stripped by this function.

    .. warning:: This function only for use within Sphinx..

    __ https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    return _ANSI_CODES.sub('', text)
