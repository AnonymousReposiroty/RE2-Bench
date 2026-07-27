from . import _api, _version, cbook, _docstring, rcsetup

def _replacer(data, value):
    """
    Either returns ``data[value]`` or passes ``data`` back, converting any
    ``MappingView`` to a sequence.
    """
    try:
        # if key isn't a string don't bother
        if isinstance(value, str):
            # try to use __getitem__
            value = data[value]
    except Exception:
        # key does not exist, silently fall back to key
        pass
    return cbook.sanitize_sequence(value)
