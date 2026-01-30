import numpy as np
from io import StringIO
def string2Array(s):
    s = s.strip()

    # String case (elements quoted, no commas, arbitrary newlines)
    if "'" in s or '"' in s:
        # Remove brackets & quotes, collapse all whitespace to single spaces,
        # then parse as a single whitespace-delimited row of strings.
        payload = (
            s.replace('[', ' ').replace(']', ' ')
             .replace("'", ' ').replace('"', ' ')
        )
        payload = ' '.join(payload.split())
        arr = np.loadtxt(StringIO(payload), dtype=str, comments=None)
        return np.atleast_1d(arr)

    # Numeric case (works for 1D or 2D pretty-printed arrays)
    txt = s.replace('[', '').replace(']', '')
    try:
        return np.loadtxt(StringIO(txt), dtype=float, comments=None)
    except ValueError:
        # If wrapping caused uneven rows, flatten newlines to a single row
        txt_one_line = ' '.join(txt.split())
        return np.loadtxt(StringIO(txt_one_line), dtype=float, comments=None)

# Problem: scikit-learn__scikit-learn-11333@@sklearn.utils.metaestimators.py@@_set_params
# Benchmark: Swebench
# Module: sklearn.utils.metaestimators
# Function: _set_params

from sklearn.utils.metaestimators import _BaseComposition

class DummyComposition(_BaseComposition):
    def __init__(self):
        self.transformers = []
        self.remainder = 'passthrough'
        self.n_jobs = 1
        self.transformer_weights = None
        self.transformers_ = self.transformers
        self._remainder = ('remainder', 'passthrough', None)    
def test_input(pred_input):
    obj_ins = DummyComposition()
    obj_ins.transformers = None
    obj_ins.remainder = 'drop'
    obj_ins.n_jobs = 1
    obj_ins.transformer_weights = None
    obj_ins_pred = DummyComposition()
    obj_ins_pred.transformers = pred_input['self']['transformers']
    obj_ins_pred.remainder = pred_input['self']['remainder']
    obj_ins_pred.n_jobs = pred_input['self']['n_jobs']
    obj_ins_pred.transformer_weights = pred_input['self']['transformer_weights']
    assert obj_ins._set_params(attr = '_transformers', trans1__with_mean = False)==obj_ins_pred._set_params(attr = pred_input['args']['attr'], trans1__with_mean = pred_input['kwargs']['trans1__with_mean']), 'Prediction failed!'