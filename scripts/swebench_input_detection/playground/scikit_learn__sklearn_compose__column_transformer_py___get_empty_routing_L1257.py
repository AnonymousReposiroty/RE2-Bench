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

# Problem: scikit-learn@@sklearn_compose__column_transformer.py@@_get_empty_routing_L1257
# Module: sklearn.compose._column.transformer
# Function: _get_empty_routing
# Line: 1257

from sklearn.compose import ColumnTransformer


def test_input(pred_input):
    obj_ins = ColumnTransformer(transformers = [['trans', 'Trans()', '<function test_column_transformer.<locals>.<lambda> at 0x70e5fc5df600>']], remainder = 'drop', sparse_threshold = 0.3, n_jobs = None, transformer_weights = None, verbose = False, verbose_feature_names_out = True)
    obj_ins.force_int_remainder_cols = 'deprecated'
    obj_ins.n_features_in_ = 2
    obj_ins._columns = [0]
    obj_ins._transformer_to_input_indices = {'trans': [0], 'remainder': [1]}
    obj_ins._remainder = ['remainder', 'drop', [1]]
    obj_ins_pred = ColumnTransformer(transformers = pred_input['self']['transformers'], remainder = pred_input['self']['remainder'], sparse_threshold = pred_input['self']['sparse_threshold'], n_jobs = pred_input['self']['n_jobs'], transformer_weights = pred_input['self']['transformer_weights'], verbose = pred_input['self']['verbose'], verbose_feature_names_out = pred_input['self']['verbose_feature_names_out'])
    obj_ins_pred.force_int_remainder_cols = pred_input['self']['force_int_remainder_cols']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._columns = pred_input['self']['_columns']
    obj_ins_pred._transformer_to_input_indices = pred_input['self']['_transformer_to_input_indices']
    obj_ins_pred._remainder = pred_input['self']['_remainder']
    assert obj_ins._get_empty_routing()==obj_ins_pred._get_empty_routing(), 'Prediction failed!'


