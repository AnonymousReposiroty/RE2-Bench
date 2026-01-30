from matplotlib import _api, _c_internal_utils, mlab

def _local_over_kwdict(
        local_var, kwargs, *keys,
        warning_cls=_api.MatplotlibDeprecationWarning):
    out = local_var
    for key in keys:
        kwarg_val = kwargs.pop(key, None)
        if kwarg_val is not None:
            if out is None:
                out = kwarg_val
            else:
                _api.warn_external(f'"{key}" keyword argument will be ignored',
                                   warning_cls)
    return out
