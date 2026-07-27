import re

def _unquote(v):
    if v[:1] in ('"', "'"):
        return re.sub(r'\\([0-9]{1,3}|u[0-9a-f]{4}|.)', _escape_sub_callback,
                      v[1:-1])
    elif v in ('?', ''):
        return None
    else:
        return v
