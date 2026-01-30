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

from sklearn.decomposition import dict_learning

def test_input(pred_input):
    X = np.array(
        [
        [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799, -0.97727788, 0.95008842, -0.15135721],
        [-0.10321885, 0.4105985, 0.14404357, 1.45427351, 0.76103773, 0.12167502, 0.44386323, 0.33367433],
        [1.49407907, -0.20515826, 0.3130677, -0.85409574, -2.55298982, 0.6536186, 0.8644362, -0.74216502],
        [2.26975462, -1.45436567, 0.04575852, -0.18718385, 1.53277921, 1.46935877, 0.15494743, 0.37816252],
        [-0.88778575, -1.98079647, -0.34791215, 0.15634897, 1.23029068, 1.20237985, -0.38732682, -0.30230275],
        [-1.04855297, -1.42001794, -1.70627019, 1.9507754, -0.50965218, -0.4380743, -1.25279536, 0.77749036],
        [-1.61389785, -0.21274028, -0.89546656, 0.3869025, -0.51080514, -1.18063218, -0.02818223, 0.42833187],
        [0.06651722, 0.3024719, -0.63432209, -0.36274117, -0.67246045, -0.35955316, -0.81314628, -1.7262826],
        [0.17742614, -0.40178094, -1.63019835, 0.46278226, -0.90729836, 0.0519454, 0.72909056, 0.12898291],
        [1.13940068, -1.23482582, 0.40234164, -0.68481009, -0.87079715, -0.57884966, -0.31155253, 0.05616534]
        ]
    )
    assert dict_learning(X = X, n_components = 5, alpha = 1, tol = 1e-08, max_iter = 1000, method = 'cd', n_jobs = 1, code_init = None, dict_init = None, verbose = False, return_n_iter = True) == dict_learning(X = pred_input['args']['X'], n_components = pred_input['args']['n_components'], alpha = pred_input['args']['alpha'], tol = pred_input['kwargs']['tol'], max_iter = pred_input['kwargs']['max_iter'], method = pred_input['kwargs']['method'], method_max_iter = pred_input['kwargs']['method_max_iter'], n_jobs = pred_input['kwargs']['n_jobs'], code_init = pred_input['kwargs']['code_init'], dict_init = pred_input['kwargs']['dict_init'], verbose = pred_input['kwargs']['verbose'], random_state = pred_input['kwargs']['random_state'], return_n_iter = pred_input['kwargs']['return_n_iter']), 'Prediction failed!'

