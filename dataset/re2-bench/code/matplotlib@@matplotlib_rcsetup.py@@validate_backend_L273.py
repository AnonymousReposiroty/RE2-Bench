from matplotlib.backends import backend_registry

def validate_backend(s):
    if s is _auto_backend_sentinel or backend_registry.is_valid_backend(s):
        return s
    else:
        msg = (f"'{s}' is not a valid value for backend; supported values are "
               f"{backend_registry.list_all()}")
        raise ValueError(msg)
