def _validate_marker(s):
    try:
        return validate_int(s)
    except ValueError as e:
        try:
            return validate_string(s)
        except ValueError as e:
            raise ValueError('Supported markers are [string, int]') from e
