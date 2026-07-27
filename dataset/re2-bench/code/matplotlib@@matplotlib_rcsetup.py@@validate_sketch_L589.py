def validate_sketch(s):

    if isinstance(s, str):
        s = s.lower().strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
    if s == 'none' or s is None:
        return None
    try:
        return tuple(_listify_validator(validate_float, n=3)(s))
    except ValueError as exc:
        raise ValueError("Expected a (scale, length, randomness) tuple") from exc
