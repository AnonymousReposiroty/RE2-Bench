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

from sklearn.preprocessing import StandardScaler


def test_input(pred_input):
	obj_ins = StandardScaler(with_mean = True, with_std = True, copy = True)
	obj_ins_pred = StandardScaler(with_mean = pred_input['self']['with_mean'], with_std = pred_input['self']['with_std'], copy = pred_input['self']['copy'])
	assert obj_ins.partial_fit(X = string2Array('[[  0.07502744  -4.56321133   1.76919009   5.35247043   0.30407403\n    4.63432284  -2.34784453  -0.08644247   1.96216865   3.61707997\n    6.27656227   1.97693625 -10.78447334  -1.73408131   6.80635231\n   -3.77705996   5.47769956   3.77950018  -2.24260137  -4.21340792\n    1.23239203   1.41069131  -4.15420605  -1.51097566  -0.67638104\n   -0.20541441   0.9497847   -0.49377556  15.01430105  -1.10385377]]'))==obj_ins_pred.partial_fit(X = pred_input['args']['X']), 'Prediction failed!'
