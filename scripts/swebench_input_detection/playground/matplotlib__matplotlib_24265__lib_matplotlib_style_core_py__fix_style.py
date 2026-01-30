import contextlib
import logging
import os
from pathlib import Path
import re
import warnings
import matplotlib as mpl
from matplotlib import _api
_log = logging.getLogger(__name__)
__all__ = ['use', 'context', 'available', 'library', 'reload_library']
BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')
USER_LIBRARY_PATHS = [os.path.join(mpl.get_configdir(), 'stylelib')]
STYLE_EXTENSION = 'mplstyle'
STYLE_BLACKLIST = {'interactive', 'backend', 'webagg.port', 'webagg.address', 'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback', 'toolbar', 'timezone', 'figure.max_open_warning', 'figure.raise_window', 'savefig.directory', 'tk.window_focus', 'docstring.hardcopy', 'date.epoch'}
_DEPRECATED_SEABORN_STYLES = {s: s.replace('seaborn', 'seaborn-v0_8') for s in ['seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-dark-palette', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid']}
_DEPRECATED_SEABORN_MSG = "The seaborn styles shipped by Matplotlib are deprecated since %(since)s, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead."


available = []
style_alias = {'mpl20': 'default', 'mpl15': 'classic'}
def fix_style(s):
    if isinstance(s, str):
        s = style_alias.get(s, s)
        if s in _DEPRECATED_SEABORN_STYLES:
            _api.warn_deprecated('3.6', message=_DEPRECATED_SEABORN_MSG)
            s = _DEPRECATED_SEABORN_STYLES[s]
    return s

def test_input(pred_input):
	assert fix_style(s = 'default')==fix_style(s = pred_input['args']['s']), 'Prediction failed!'
