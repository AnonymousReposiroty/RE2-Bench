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

# Problem: scikit-learn@@sklearn_pipeline.py@@__sklearn_tags___L1152
# Module: sklearn.pipeline
# Function: __sklearn_tags__
# Line: 1152

from sklearn.pipeline import Pipeline


def test_input(pred_input):
    obj_ins = Pipeline(steps = [['kernel_pca', "KernelPCA(gamma=np.float64(0.25), kernel='rbf', n_components=2)"], ['Perceptron', 'Perceptron(max_iter=5)']], transform_input = None, memory = None, verbose = False)
    obj_ins_pred = Pipeline(steps = pred_input['self']['steps'], transform_input = pred_input['self']['transform_input'], memory = pred_input['self']['memory'], verbose = pred_input['self']['verbose'])
    assert obj_ins.__sklearn_tags__()==obj_ins_pred.__sklearn_tags__(), 'Prediction failed!'