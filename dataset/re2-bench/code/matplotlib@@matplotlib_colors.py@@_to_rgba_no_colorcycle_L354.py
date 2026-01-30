from numbers import Real
import re
import numpy as np

def _to_rgba_no_colorcycle(c, alpha=None):
    """
    Convert *c* to an RGBA color, with no support for color-cycle syntax.

    If *alpha* is given, force the alpha value of the returned RGBA tuple
    to *alpha*. Otherwise, the alpha value from *c* is used, if it has alpha
    information, or defaults to 1.

    *alpha* is ignored for the color value ``"none"`` (case-insensitive),
    which always maps to ``(0, 0, 0, 0)``.
    """
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("'alpha' must be between 0 and 1, inclusive")
    orig_c = c
    if c is np.ma.masked:
        return (0., 0., 0., 0.)
    if isinstance(c, str):
        if c.lower() == "none":
            return (0., 0., 0., 0.)
        # Named color.
        try:
            # This may turn c into a non-string, so we check again below.
            c = _colors_full_map[c]
        except KeyError:
            if len(c) != 1:
                try:
                    c = _colors_full_map[c.lower()]
                except KeyError:
                    pass
    if isinstance(c, str):
        if re.fullmatch("#[a-fA-F0-9]+", c):
            if len(c) == 7:  # #rrggbb hex format.
                return (*[n / 0xff for n in bytes.fromhex(c[1:])],
                        alpha if alpha is not None else 1.)
            elif len(c) == 4:  # #rgb hex format, shorthand for #rrggbb.
                return (*[int(n, 16) / 0xf for n in c[1:]],
                        alpha if alpha is not None else 1.)
            elif len(c) == 9:  # #rrggbbaa hex format.
                color = [n / 0xff for n in bytes.fromhex(c[1:])]
                if alpha is not None:
                    color[-1] = alpha
                return tuple(color)
            elif len(c) == 5:  # #rgba hex format, shorthand for #rrggbbaa.
                color = [int(n, 16) / 0xf for n in c[1:]]
                if alpha is not None:
                    color[-1] = alpha
                return tuple(color)
            else:
                raise ValueError(f"Invalid hex color specifier: {orig_c!r}")
        # string gray.
        try:
            c = float(c)
        except ValueError:
            pass
        else:
            if not (0 <= c <= 1):
                raise ValueError(
                    f"Invalid string grayscale value {orig_c!r}. "
                    f"Value must be within 0-1 range")
            return c, c, c, alpha if alpha is not None else 1.
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # turn 2-D array into 1-D array
    if isinstance(c, np.ndarray):
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.reshape(-1)
    # tuple color.
    if not np.iterable(c):
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    if len(c) not in [3, 4]:
        raise ValueError("RGBA sequence should have length 3 or 4")
    if not all(isinstance(x, Real) for x in c):
        # Checks that don't work: `map(float, ...)`, `np.array(..., float)` and
        # `np.array(...).astype(float)` would all convert "0.5" to 0.5.
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # Return a tuple to prevent the cached value from being modified.
    c = tuple(map(float, c))
    if len(c) == 3 and alpha is None:
        alpha = 1
    if alpha is not None:
        c = c[:3] + (alpha,)
    if any(elem < 0 or elem > 1 for elem in c):
        raise ValueError("RGBA values should be within 0-1 range")
    return c
