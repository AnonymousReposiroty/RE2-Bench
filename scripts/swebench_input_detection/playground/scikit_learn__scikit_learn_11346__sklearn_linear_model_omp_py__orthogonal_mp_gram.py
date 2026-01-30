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

from sklearn.linear_model import orthogonal_mp_gram

def test_input(pred_input):
	assert orthogonal_mp_gram(Gram = string2Array('[[ 1.00000000e+00  3.81506944e-01  1.14715887e-01 -4.68607043e-01\n   5.84719658e-02 -1.00601972e-01 -1.83510043e-01 -1.65667971e-02\n   3.27876822e-01 -4.48139501e-02 -1.62726917e-02  4.96932416e-01]\n [ 3.81506944e-01  1.00000000e+00  5.98382782e-02  9.25732329e-02\n  -2.06897998e-01 -8.91322682e-02 -3.17686864e-01  2.65142869e-02\n  -3.54909210e-01 -7.21809509e-01  9.00883273e-03  6.58684170e-01]\n [ 1.14715887e-01  5.98382782e-02  1.00000000e+00 -1.49677007e-01\n   1.35940238e-01 -4.37486996e-01 -3.09998023e-01 -5.04662967e-01\n   2.97667021e-01 -3.39423421e-01 -1.46652624e-01  6.71427599e-03]\n [-4.68607043e-01  9.25732329e-02 -1.49677007e-01  1.00000000e+00\n  -3.90373744e-01  2.40310978e-01  4.07849862e-01  3.92318210e-01\n  -7.21266184e-01 -1.34785291e-01 -2.76100508e-01 -2.79362559e-04]\n [ 5.84719658e-02 -2.06897998e-01  1.35940238e-01 -3.90373744e-01\n   1.00000000e+00 -7.46501844e-01 -1.20107166e-01 -6.24618127e-01\n   5.78463889e-01  3.10437036e-01 -7.78310684e-02 -4.36706827e-02]\n [-1.00601972e-01 -8.91322682e-02 -4.37486996e-01  2.40310978e-01\n  -7.46501844e-01  1.00000000e+00  2.67422100e-02  3.54147450e-01\n  -3.43679149e-01  2.78389735e-02  3.22626373e-02 -3.91819703e-01]\n [-1.83510043e-01 -3.17686864e-01 -3.09998023e-01  4.07849862e-01\n  -1.20107166e-01  2.67422100e-02  1.00000000e+00  7.37160571e-01\n  -2.73861456e-01  3.44858307e-01 -1.42671753e-01 -1.14706234e-01]\n [-1.65667971e-02  2.65142869e-02 -5.04662967e-01  3.92318210e-01\n  -6.24618127e-01  3.54147450e-01  7.37160571e-01  1.00000000e+00\n  -5.23514292e-01  7.13866157e-02  5.21231760e-02  2.17745098e-01]\n [ 3.27876822e-01 -3.54909210e-01  2.97667021e-01 -7.21266184e-01\n   5.78463889e-01 -3.43679149e-01 -2.73861456e-01 -5.23514292e-01\n   1.00000000e+00  6.55590151e-02 -3.79448801e-01 -3.46617185e-01]\n [-4.48139501e-02 -7.21809509e-01 -3.39423421e-01 -1.34785291e-01\n   3.10437036e-01  2.78389735e-02  3.44858307e-01  7.13866157e-02\n   6.55590151e-02  1.00000000e+00  4.37528732e-01 -1.70499778e-01]\n [-1.62726917e-02  9.00883273e-03 -1.46652624e-01 -2.76100508e-01\n  -7.78310684e-02  3.22626373e-02 -1.42671753e-01  5.21231760e-02\n  -3.79448801e-01  4.37528732e-01  1.00000000e+00  3.73148348e-01]\n [ 4.96932416e-01  6.58684170e-01  6.71427599e-03 -2.79362559e-04\n  -4.36706827e-02 -3.91819703e-01 -1.14706234e-01  2.17745098e-01\n  -3.46617185e-01 -1.70499778e-01  3.73148348e-01  1.00000000e+00]]'), Xy = string2Array('[[ 0.1260535 ]\n [-0.4460294 ]\n [ 0.29305911]\n [-0.84156526]\n [ 2.15579371]\n [-1.60930398]\n [-0.25892627]\n [-1.34654783]\n [ 1.24704881]\n [ 0.66923821]\n [-0.16778773]\n [-0.09414498]]'), n_nonzero_coefs = 1, tol = None, norms_squared = string2Array('[4.64744652]'), copy_Xy = False)==orthogonal_mp_gram(Gram = pred_input['kwargs']['Gram'], Xy = pred_input['kwargs']['Xy'], n_nonzero_coefs = pred_input['kwargs']['n_nonzero_coefs'], tol = pred_input['kwargs']['tol'], norms_squared = pred_input['kwargs']['norms_squared'], copy_Xy = pred_input['kwargs']['copy_Xy']), 'Prediction failed!'
