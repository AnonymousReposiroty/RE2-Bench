def _could_extract_minus_sign(expr):
    # assume expr is Add-like
    # We choose the one with less arguments with minus signs
    negative_args = sum(1 for i in expr.args
        if i.could_extract_minus_sign())
    positive_args = len(expr.args) - negative_args
    if positive_args > negative_args:
        return False
    elif positive_args < negative_args:
        return True
    # choose based on .sort_key() to prefer
    # x - 1 instead of 1 - x and
    # 3 - sqrt(2) instead of -3 + sqrt(2)
    return bool(expr.sort_key() < (-expr).sort_key())
