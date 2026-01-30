def _has_alpha_channel(c):
    """
    Return whether *c* is a color with an alpha channel.

    If *c* is not a valid color specifier, then the result is undefined.
    """
    # The following logic uses the assumption that c is a valid color spec.
    # For speed and simplicity, we intentionally don't care about other inputs.
    # Anything can happen with them.

    # if c is a hex, it has an alpha channel when it has 4 (or 8) digits after '#'
    if isinstance(c, str):
        if c[0] == '#' and (len(c) == 5 or len(c) == 9):
            # example: '#fff8' or '#0f0f0f80'
            return True
    else:
        # if c isn't a string, it can be an RGB(A) or a color-alpha tuple
        # if it has length 4, it has an alpha channel
        if len(c) == 4:
            # example: [0.5, 0.5, 0.5, 0.5]
            return True

        # if it has length 2, it's a color/alpha tuple
        # if the second element isn't None or the first element has length = 4
        if len(c) == 2 and (c[1] is not None or _has_alpha_channel(c[0])):
            # example: ([0.5, 0.5, 0.5, 0.5], None) or ('r', 0.5)
            return True

    # otherwise it doesn't have an alpha channel
    return False
