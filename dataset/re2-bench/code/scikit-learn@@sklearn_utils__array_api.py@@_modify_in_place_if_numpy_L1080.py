def _modify_in_place_if_numpy(xp, func, *args, out=None, **kwargs):
    if _is_numpy_namespace(xp):
        func(*args, out=out, **kwargs)
    else:
        out = func(*args, **kwargs)
    return out
