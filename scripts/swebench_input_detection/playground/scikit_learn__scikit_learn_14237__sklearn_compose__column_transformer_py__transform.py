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

# Problem: scikit-learn__scikit-learn-14237@@sklearn.compose._column_transformer.py@@transform
# Benchmark: Swebench
# Module: sklearn.compose._column_transformer
# Function: transform

from sklearn.compose._column_transformer import ColumnTransformer


def test_input(pred_input):
    obj_ins = ColumnTransformer(transformers = None, remainder = 'drop', sparse_threshold = 0.3, n_jobs = None, transformer_weights = None, verbose = False)
    obj_ins._columns = None
    obj_ins._n_features = 3
    obj_ins._remainder = ['remainder', 'drop', None]
    obj_ins.sparse_output_ = False
    obj_ins.transformers_ = []
    obj_ins_pred = ColumnTransformer(transformers = pred_input['self']['transformers'], remainder = pred_input['self']['remainder'], sparse_threshold = pred_input['self']['sparse_threshold'], n_jobs = pred_input['self']['n_jobs'], transformer_weights = pred_input['self']['transformer_weights'], verbose = pred_input['self']['verbose'])
    obj_ins_pred._columns = pred_input['self']['_columns']
    obj_ins_pred._n_features = pred_input['self']['_n_features']
    obj_ins_pred._remainder = pred_input['self']['_remainder']
    obj_ins_pred.sparse_output_ = pred_input['self']['sparse_output_']
    obj_ins_pred.transformers_ = pred_input['self']['transformers_']
    assert obj_ins.transform(X = [[0],[0]])==obj_ins_pred.transform(X = pred_input['args']['X']), 'Prediction failed!'
    