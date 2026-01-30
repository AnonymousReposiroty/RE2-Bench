import os

def _update_user_library(library):
    """Update style library with user-defined rc files."""
    for stylelib_path in map(os.path.expanduser, USER_LIBRARY_PATHS):
        styles = _read_style_directory(stylelib_path)
        _update_nested_dict(library, styles)
    return library
