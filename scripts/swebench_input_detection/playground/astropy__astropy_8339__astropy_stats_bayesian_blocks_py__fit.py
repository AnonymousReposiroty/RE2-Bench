from astropy.stats.bayesian_blocks import FitnessFunc
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


def test_input(pred_input):
	obj_ins = FitnessFunc(p0 = 0.05, gamma = None, ncp_prior = None)
	obj_ins_pred = FitnessFunc(p0 = pred_input['self']['p0'], gamma = pred_input['self']['gamma'], ncp_prior = pred_input['self']['ncp_prior'])
	assert obj_ins.fit(t = string2Array('[66.89240597 86.41675651 23.01852682 49.91933799 57.20041992 76.85540143\n  4.36037718 99.45505108 46.9944514  27.95603418 88.34940223 74.77187739\n 95.3071847  33.07503047 55.27649668 57.22924692 98.03315837  7.5346256\n 30.56970193 19.09110312]'), x = string2Array('[-0.0111204  -0.17490694 -0.13414035  0.83458264 -0.114772    0.13896028\n -0.08552405 -0.10270618  0.09920921 -0.04493044  0.04186395 -0.07281792\n -0.1801888   0.04677171 -0.15514249 -0.05104932 -0.07329511  0.01953465\n -0.05559449 -0.01277348]'), sigma = 0.1)==obj_ins_pred.fit(t = pred_input['args']['t'], x = pred_input['args']['x'], sigma = pred_input['args']['sigma']), 'Prediction failed!'
