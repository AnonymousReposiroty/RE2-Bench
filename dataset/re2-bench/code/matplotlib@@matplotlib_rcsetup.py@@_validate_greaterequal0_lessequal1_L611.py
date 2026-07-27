def _validate_greaterequal0_lessequal1(s):
    s = validate_float(s)
    if 0 <= s <= 1:
        return s
    else:
        raise RuntimeError(f'Value must be >=0 and <=1; got {s}')
