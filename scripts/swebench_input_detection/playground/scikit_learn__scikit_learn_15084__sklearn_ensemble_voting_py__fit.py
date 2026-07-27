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

from sklearn.ensemble._voting import _BaseVoting


def test_input(pred_input):
	obj_ins = _BaseVoting(estimators = None, weights = None, n_jobs = None)
	obj_ins_pred = _BaseVoting(estimators = pred_input['self']['estimators'], weights = pred_input['self']['weights'], n_jobs = pred_input['self']['n_jobs'])
	assert obj_ins.fit(X = {'_shape': [40, 10], 'maxprint': 50, 'indices': '[7 8 3 7 9 0 7 8 2 6 8 0 2 9 8 3 9 1 4 6 8 2 0 3 4 7 9 1 5 6 3 4 5 7 4 3 9\n 0 9 1 5 8 0 1 4 8 3 0 1 3 5 6 9 1 6 2 7 8 9 0 8 9 9 1 4 8 7 0 2 4 5 6 3 5]', 'indptr': '[ 0  2  5  7  8  8  9 11 13 14 15 17 21 22 22 27 30 33 34 35 36 37 39 42\n 42 43 46 47 53 55 59 60 61 61 61 62 63 66 67 72 74]', 'data': '[[[0.891773  ]]\n\n [[0.96366276]]\n\n [[0.92559664]]\n\n [[0.83261985]]\n\n [[0.87001215]]\n\n [[0.97861834]]\n\n [[0.94466892]]\n\n [[0.94374808]]\n\n [[0.98837384]]\n\n [[0.82099323]]\n\n [[0.83794491]]\n\n [[0.97645947]]\n\n [[0.97676109]]\n\n [[0.9292962 ]]\n\n [[0.82894003]]\n\n [[0.96218855]]\n\n [[0.95274901]]\n\n [[0.84640867]]\n\n [[0.81379782]]\n\n [[0.8811032 ]]\n\n [[0.88173536]]\n\n [[0.95608363]]\n\n [[0.8965466 ]]\n\n [[0.89192336]]\n\n [[0.80619399]]\n\n [[0.91948261]]\n\n [[0.99884701]]\n\n [[0.86812606]]\n\n [[0.84800823]]\n\n [[0.80731896]]\n\n [[0.86638233]]\n\n [[0.97552151]]\n\n [[0.85580334]]\n\n [[0.92808129]]\n\n [[0.934214  ]]\n\n [[0.94437239]]\n\n [[0.89667129]]\n\n [[0.99033895]]\n\n [[0.83104846]]\n\n [[0.87265066]]\n\n [[0.95279166]]\n\n [[0.94737059]]\n\n [[0.86385561]]\n\n [[0.94043195]]\n\n [[0.90371974]]\n\n [[0.96193638]]\n\n [[0.92952932]]\n\n [[0.97749514]]\n\n [[0.87650525]]\n\n [[0.96157015]]\n\n [[0.94931882]]\n\n [[0.9413777 ]]\n\n [[0.87428797]]\n\n [[0.84894356]]\n\n [[0.98182939]]\n\n [[0.82211773]]\n\n [[0.86219152]]\n\n [[0.97291949]]\n\n [[0.96083466]]\n\n [[0.9065555 ]]\n\n [[0.86055117]]\n\n [[0.95943334]]\n\n [[0.95640572]]\n\n [[0.90398395]]\n\n [[0.88204141]]\n\n [[0.90404439]]\n\n [[0.9591666 ]]\n\n [[0.85772264]]\n\n [[0.95187448]]\n\n [[0.82076712]]\n\n [[0.90884372]]\n\n [[0.81552382]]\n\n [[0.84903831]]\n\n [[0.95898272]]]'}, y = string2Array('[1 3 0 3 3 1 1 0 2 0 0 1 0 1 0 1 0 0 0 2 3 3 1 0 2 1 3 0 3 1 0 2 0 2 2 1 2\n 1 2 1]'), sample_weight = None)==obj_ins_pred.fit(X = pred_input['args']['X'], y = pred_input['args']['y'], sample_weight = pred_input['args']['sample_weight']), 'Prediction failed!'