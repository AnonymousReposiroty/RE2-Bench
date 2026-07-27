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

# Problem: scikit-learn@@sklearn_cluster__dbscan.py@@fit_predict_L476
# Module: sklearn.cluster._dbscan
# Function: fit_predict
# Line: 476

from sklearn.cluster._dbscan import DBSCAN


def test_input(pred_input):
    obj_ins = DBSCAN(eps = 0.5, min_samples = 5, metric = 'euclidean', metric_params = None, algorithm = 'auto', leaf_size = 30, p = None, n_jobs = None)
    obj_ins_pred = DBSCAN(eps = pred_input['self']['eps'], min_samples = pred_input['self']['min_samples'], metric = pred_input['self']['metric'], metric_params = pred_input['self']['metric_params'], algorithm = pred_input['self']['algorithm'], leaf_size = pred_input['self']['leaf_size'], p = pred_input['self']['p'], n_jobs = pred_input['self']['n_jobs'])
    s = '[[11.70562094, 11.16006288],\n       [11.39149519, 11.89635728],\n       [11.7470232 , 10.60908885],\n       [11.38003537, 10.93945712],\n       [10.95871246, 11.1642394 ],\n       [11.05761743, 11.5817094 ],\n       [11.30441509, 11.04867001],\n       [11.17754529, 11.13346973],\n       [11.59763163, 10.91793669],\n       [11.12522708, 10.6583617 ],\n       [ 9.97880407, 11.26144744],\n       [11.34577448, 10.70313399],\n       [11.90790185, 10.41825373],\n       [11.01830341, 10.92512646],\n       [11.61311169, 11.58774351],\n       [11.06197897, 11.15126501],\n       [10.6448857 , 10.20768141],\n       [10.86083514, 11.06253959],\n       [11.49211627, 11.48095194],\n       [10.84506927, 10.8790789 ],\n       [ 8.58057881,  8.43199283],\n       [ 8.31749192,  9.78031016],\n       [ 8.79613913,  8.82477028],\n       [ 8.49888186,  9.31099614],\n       [ 8.35444086,  8.91490389],\n       [ 8.64181338,  9.154761  ],\n       [ 8.79567794,  8.52774713],\n       [ 8.98872711,  9.17133275],\n       [ 9.02660689,  9.12098876],\n       [ 8.74627116,  8.85490353],\n       [ 8.73101582,  8.85617874],\n       [ 8.67474149,  8.30948696],\n       [ 9.07097046,  8.83928763],\n       [ 8.34792066,  9.1851129 ],\n       [ 8.63708065,  9.02077816],\n       [ 9.29163622,  9.05159316],\n       [ 9.45576027,  8.50606967],\n       [ 9.16093666,  8.72607596],\n       [ 8.65168114,  8.76846013],\n       [ 8.87537899,  9.02246614],\n       [10.53394006,  9.36033059],\n       [11.18626498,  8.38550253],\n       [11.59530088,  9.75835567],\n       [11.47151183,  8.92803007],\n       [10.57169895,  9.42178069],\n       [10.83872922,  9.48897803],\n       [11.08330999,  9.39065561],\n       [11.14254656,  9.28262927],\n       [11.00420001,  9.7143482 ],\n       [11.05076484,  9.16079575],\n       [11.75326028,  8.46089638],\n       [10.491806  ,  9.38775868],\n       [10.53075064,  9.77744847],\n       [10.83455241,  8.70101808],\n       [11.76917681,  9.59220592],\n       [11.74702358,  9.36241786],\n       [10.65550973,  9.76402598],\n       [10.89279865,  9.32098256],\n       [11.37890079,  8.93799596],\n       [11.24563175,  9.36888267]]'
    # Convert string → numpy array
    arr = np.array(eval(s))
    a = obj_ins.fit_predict(X = arr, y = None, sample_weight = np.array([3, 4, 2, 4, 4, 1, 2, 2, 2, 4, 3, 2, 4, 1, 3, 1, 3, 4, 0, 3, 1, 4,3, 0, 0, 2, 2, 1, 3, 3, 2, 3, 3, 0, 2, 4, 2, 4, 0, 1, 3, 0, 3, 1,1, 0, 1, 4, 1, 3, 3, 3, 3, 4, 2, 0, 3, 1, 3, 1]))
 
    assert a==obj_ins_pred.fit_predict(X = pred_input['args']['X'], y = pred_input['args']['y'], sample_weight = pred_input['args']['sample_weight']), 'Prediction failed!'