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

from sklearn.preprocessing import OneHotEncoder

def test_input(pred_input):
    obj_ins = OneHotEncoder(categories = 'auto', sparse = 'deprecated', sparse_output = True, dtype = {'__repr__': '"<function __repr__>"', '__hash__': '"<function __hash__>"', '__str__': '"<function __str__>"', '__lt__': '"<function __lt__>"', '__le__': '"<function __le__>"', '__eq__': '"<function __eq__>"', '__ne__': '"<function __ne__>"', '__gt__': '"<function __gt__>"', '__ge__': '"<function __ge__>"', '__add__': '"<function __add__>"', '__radd__': '"<function __radd__>"', '__sub__': '"<function __sub__>"', '__rsub__': '"<function __rsub__>"', '__mul__': '"<function __mul__>"', '__rmul__': '"<function __rmul__>"', '__mod__': '"<function __mod__>"', '__rmod__': '"<function __rmod__>"', '__divmod__': '"<function __divmod__>"', '__rdivmod__': '"<function __rdivmod__>"', '__pow__': '"<function __pow__>"', '__rpow__': '"<function __rpow__>"', '__neg__': '"<function __neg__>"', '__pos__': '"<function __pos__>"', '__abs__': '"<function __abs__>"', '__bool__': '"<function __bool__>"', '__int__': '"<function __int__>"', '__float__': '"<function __float__>"', '__floordiv__': '"<function __floordiv__>"', '__rfloordiv__': '"<function __rfloordiv__>"', '__truediv__': '"<function __truediv__>"', '__rtruediv__': '"<function __rtruediv__>"', '__new__': '"<function __new__>"', 'as_integer_ratio': '"<function as_integer_ratio>"', '__doc__': None}, handle_unknown = 'error', drop = None, min_frequency = None, max_categories = None, feature_name_combiner = 'concat')
    obj_ins_pred = OneHotEncoder(categories = pred_input['self']['categories'], sparse = pred_input['self']['sparse'], sparse_output = pred_input['self']['sparse_output'], dtype = pred_input['self']['dtype'], handle_unknown = pred_input['self']['handle_unknown'], drop = pred_input['self']['drop'], min_frequency = pred_input['self']['min_frequency'], max_categories = pred_input['self']['max_categories'], feature_name_combiner = pred_input['self']['feature_name_combiner'])
    obj_ins._infrequent_enabled = False
    obj_ins_pred._infrequent_enabled = pred_input['self']['_infrequent_enabled']
    assert obj_ins._map_drop_idx_to_infrequent(feature_idx = 2, drop_idx = '1')==obj_ins_pred._map_drop_idx_to_infrequent(feature_idx = pred_input['args']['feature_idx'], drop_idx = pred_input['args']['drop_idx']), 'Prediction failed!'