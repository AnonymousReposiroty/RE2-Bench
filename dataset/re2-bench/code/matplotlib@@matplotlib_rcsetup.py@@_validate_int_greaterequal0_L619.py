def _validate_int_greaterequal0(s):
    s = validate_int(s)
    if s >= 0:
        return s
    else:
        raise RuntimeError(f'Value must be >=0; got {s}')
