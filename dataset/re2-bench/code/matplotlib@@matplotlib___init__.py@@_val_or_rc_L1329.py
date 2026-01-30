def _val_or_rc(val, *rc_names):
    """
    If *val* is None, the first not-None value in ``mpl.rcParams[rc_names[i]]``.
    If all are None returns ``mpl.rcParams[rc_names[-1]]``.
    """
    if val is not None:
        return val

    for rc_name in rc_names[:-1]:
        if rcParams[rc_name] is not None:
            return rcParams[rc_name]
    return rcParams[rc_names[-1]]
