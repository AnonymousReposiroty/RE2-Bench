from . import _api, _version, cbook, _docstring, rcsetup

def rc_params_from_file(fname, fail_on_error=False, use_default_template=True):
    """
    Construct a `RcParams` from file *fname*.

    Parameters
    ----------
    fname : str or path-like
        A file with Matplotlib rc settings.
    fail_on_error : bool
        If True, raise an error when the parser fails to convert a parameter.
    use_default_template : bool
        If True, initialize with default parameters before updating with those
        in the given file. If False, the configuration class only contains the
        parameters specified in the file. (Useful for updating dicts.)
    """
    config_from_file = _rc_params_in_file(fname, fail_on_error=fail_on_error)

    if not use_default_template:
        return config_from_file

    with _api.suppress_matplotlib_deprecation_warning():
        config = RcParams({**rcParamsDefault, **config_from_file})

    if "".join(config['text.latex.preamble']):
        _log.info("""
*****************************************************************
You have the following UNSUPPORTED LaTeX preamble customizations:
%s
Please do not ask for support with these customizations active.
*****************************************************************
""", '\n'.join(config['text.latex.preamble']))
    _log.debug('loaded rc file %s', fname)

    return config
