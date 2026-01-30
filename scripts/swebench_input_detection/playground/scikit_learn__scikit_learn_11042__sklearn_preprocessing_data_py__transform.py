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
	obj_ins = OneHotEncoder(categorical_features = 'all', sparse = True, handle_unknown = 'ignore')
	obj_ins_pred = OneHotEncoder(categorical_features = pred_input['self']['categorical_features'], sparse = pred_input['self']['sparse'], handle_unknown = pred_input['self']['handle_unknown'])
	assert obj_ins.transform(X = np.array([[4, 1, 1]]))==obj_ins_pred.transform(X = pred_input['args']['X']), 'Prediction failed!'
 
 