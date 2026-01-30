def _not_a_coeff(expr):
    """Do not treat NaN and infinities as valid polynomial coefficients. """
    if type(expr) in illegal_types or expr in finf:
        return True
    if isinstance(expr, float) and float(expr) != expr:
        return True  # nan
    return  # could be
