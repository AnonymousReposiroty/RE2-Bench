def _parse_figsize(figsize, dpi):
    """
    Convert a figsize expression to (width, height) in inches.

    Parameters
    ----------
    figsize : (float, float) or (float, float, str)
        This can be

        - a tuple ``(width, height, unit)``, where *unit* is one of "in" (inch),
          "cm" (centimenter), "px" (pixel).
        - a tuple ``(width, height)``, which is interpreted in inches, i.e. as
          ``(width, height, "in")``.

    dpi : float
        The dots-per-inch; used for converting 'px' to 'in'.
    """
    num_parts = len(figsize)
    if num_parts == 2:
        return figsize
    elif num_parts == 3:
        x, y, unit = figsize
        if unit == 'in':
            pass
        elif unit == 'cm':
            x /= 2.54
            y /= 2.54
        elif unit == 'px':
            x /= dpi
            y /= dpi
        else:
            raise ValueError(
                f"Invalid unit {unit!r} in 'figsize'; "
                "supported units are 'in', 'cm', 'px'"
            )
        return x, y
    else:
        raise ValueError(
            "Invalid figsize format, expected (x, y) or (x, y, unit) but got "
            f"{figsize!r}"
        )
