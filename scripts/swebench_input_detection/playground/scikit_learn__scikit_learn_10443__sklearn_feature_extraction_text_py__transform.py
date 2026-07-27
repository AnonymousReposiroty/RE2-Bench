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

from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sp
def recover_csr(d):
    shape = tuple(d["_shape"])
    indices = np.fromstring(d["indices"].strip("[]"), sep=" ", dtype=np.int32)
    indptr  = np.fromstring(d["indptr"].strip("[]"),  sep=" ", dtype=np.int32)
    data    = np.fromstring(d["data"].strip("[]"),    sep=" ", dtype=np.float64)  # float is fine for TF-IDF
    X = sp.csr_matrix((data, indices, indptr), shape=shape)
    # ensure canonical (sorted) format if you care about that flag
    if d.get("_has_sorted_indices", 0):
        X.sort_indices()
    return X


def test_input(pred_input):
	obj_ins = TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = False)
	obj_ins_pred = TfidfTransformer(norm = pred_input['self']['norm'], use_idf = pred_input['self']['use_idf'], smooth_idf = pred_input['self']['smooth_idf'], sublinear_tf = pred_input['self']['sublinear_tf'])
	assert obj_ins.transform(X = recover_csr({'_shape': [11, 2], 'maxprint': 50, 'indices': '[0 1 0 1 0 1 1]', 'indptr': '[0 2 4 6 7 7 7 7 7 7 7 7]', 'data': '[2 1 1 1 1 2 2]', '_has_sorted_indices': 1}))==obj_ins_pred.transform(X = recover_csr(pred_input['args']['X'])), 'Prediction failed!'
 
 
