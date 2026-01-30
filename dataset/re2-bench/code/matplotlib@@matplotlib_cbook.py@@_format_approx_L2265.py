def _format_approx(number, precision):
    """
    Format the number with at most the number of decimals given as precision.
    Remove trailing zeros and possibly the decimal point.
    """
    return f'{number:.{precision}f}'.rstrip('0').rstrip('.') or '0'
