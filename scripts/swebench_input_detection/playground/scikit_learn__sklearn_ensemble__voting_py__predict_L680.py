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

# Problem: scikit-learn@@sklearn_ensemble__voting.py@@predict_L680
# Module: sklearn.ensemble._voting
# Function: predict
# Line: 680

from sklearn.ensemble._voting import VotingRegressor


def test_input(pred_input):
    obj_ins = VotingRegressor(estimators = [['pipe1', "Pipeline(steps=[('simpleimputer', SimpleImputer()),\n                ('linearregression', LinearRegression())])"], ['pipe2', "Pipeline(steps=[('simpleimputer', SimpleImputer()),\n                ('linearregression', LinearRegression())])"]], weights = None, n_jobs = None, verbose = False)
    obj_ins.estimators_ = ["Pipeline(steps=[('simpleimputer', SimpleImputer()),\n                ('linearregression', LinearRegression())])", "Pipeline(steps=[('simpleimputer', SimpleImputer()),\n                ('linearregression', LinearRegression())])"]
    obj_ins.named_estimators_ = {'pipe1': "Pipeline(steps=[('simpleimputer', SimpleImputer()),\n                ('linearregression', LinearRegression())])", 'pipe2': "Pipeline(steps=[('simpleimputer', SimpleImputer()),\n                ('linearregression', LinearRegression())])"}
    obj_ins_pred = VotingRegressor(estimators = pred_input['self']['estimators'], weights = pred_input['self']['weights'], n_jobs = pred_input['self']['n_jobs'], verbose = pred_input['self']['verbose'])
    obj_ins_pred.estimators_ = pred_input['self']['estimators_']
    obj_ins_pred.named_estimators_ = pred_input['self']['named_estimators_']
    assert obj_ins.predict(X = np.array([[ 0.03807591, 0.05068012, 0.06169621, -0.00259226, 0.01990749, -0.01764613], [-0.00188202, -0.04464164, -0.05147406, -0.03949338, -0.06833155, -0.09220405], [ 0.08529891, 0.05068012, 0.04445121, -0.00259226, np.nan, -0.02593034], [ 0.04170844, 0.05068012, -0.01590626, -0.01107952, -0.04688253, 0.01549073], [-0.04547248, -0.04464164, 0.03906215, 0.02655962, 0.04452873, -0.02593034], [-0.04547248, -0.04464164, -0.0730303 ,  -0.03949338, np.nan, 0.00306441]]))==obj_ins_pred.predict(X = pred_input['args']['X']), 'Prediction failed!'
