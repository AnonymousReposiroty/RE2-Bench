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

from sklearn.decomposition import IncrementalPCA

def test_input(pred_input):
	obj_ins = IncrementalPCA(n_components = None, whiten = False, copy = True, batch_size = 18)
	obj_ins_pred = IncrementalPCA(n_components = pred_input['self']['n_components'], whiten = pred_input['self']['whiten'], copy = pred_input['self']['copy'], batch_size = pred_input['self']['batch_size'])
	assert obj_ins.partial_fit(X = string2Array('[[-0.03657115  0.50402514 -0.7130193 ]\n [-1.00870047  0.26292729  0.34526847]\n [-0.19518708  1.24609193 -0.75303286]\n [-1.69966675 -1.04462969 -0.54778807]\n [ 0.28537977 -0.82040101 -2.07907471]\n [ 0.497355   -0.07577611  0.60484178]\n [-1.6547215   1.39679991  1.059358  ]\n [ 0.56376607 -0.63803395  2.17641429]\n [-1.10254571 -0.70025234  0.85238413]\n [ 2.14643005  0.05494158  0.98103661]]'), check_input = False)==obj_ins_pred.partial_fit(X = pred_input['args']['X'], check_input = pred_input['kwargs']['check_input']), 'Prediction failed!'
 