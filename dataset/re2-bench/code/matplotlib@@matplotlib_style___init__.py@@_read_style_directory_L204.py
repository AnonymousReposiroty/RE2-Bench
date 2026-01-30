from pathlib import Path
import warnings
from matplotlib import _api, _docstring, rc_params_from_file, rcParamsDefault

def _read_style_directory(style_dir):
    """Return dictionary of styles defined in *style_dir*."""
    styles = dict()
    for path in Path(style_dir).glob(f"*.{_STYLE_EXTENSION}"):
        with warnings.catch_warnings(record=True) as warns:
            styles[path.stem] = rc_params_from_file(path, use_default_template=False)
        for w in warns:
            _log.warning('In %s: %s', path, w.message)
    return styles
