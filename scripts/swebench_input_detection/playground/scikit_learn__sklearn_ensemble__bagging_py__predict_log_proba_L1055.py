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

# Problem: scikit-learn@@sklearn_ensemble__bagging.py@@predict_log_proba_L1055
# Module: sklearn.ensemble._bagging
# Function: BaggingClassifier.predict_log_proba
# Line: 1055

from sklearn.ensemble._bagging import BaggingClassifier


def test_input(pred_input):
    obj_ins = BaggingClassifier(estimator = "CustomSVC(kernel='linear')", n_estimators = 10, max_samples = 1.0, max_features = 4, bootstrap = True, bootstrap_features = True, oob_score = False, warm_start = False, n_jobs = None, random_state = 1, verbose = 0)
    obj_ins.estimator_params = []
    obj_ins.n_features_in_ = 4
    obj_ins._n_samples = 112
    obj_ins.classes_ = np.array([0, 1, 2])
    obj_ins.n_classes_ = 3
    obj_ins.estimator_ = "CustomSVC(kernel='linear')"
    obj_ins._max_samples = 112
    obj_ins._max_features = 4
    obj_ins._sample_weight = None
    obj_ins.estimators_ = ["CustomSVC(kernel='linear', random_state=1028862084)", "CustomSVC(kernel='linear', random_state=870353631)", "CustomSVC(kernel='linear', random_state=788373214)", "CustomSVC(kernel='linear', random_state=1419052930)", "CustomSVC(kernel='linear', random_state=873768326)", "CustomSVC(kernel='linear', random_state=1622145301)", "CustomSVC(kernel='linear', random_state=2008179789)", "CustomSVC(kernel='linear', random_state=643033620)", "CustomSVC(kernel='linear', random_state=1357834371)", "CustomSVC(kernel='linear', random_state=1921671245)"]
    obj_ins.estimators_features_ = ['array([0, 0, 0, 2])', 'array([3, 0, 1, 1])', 'array([2, 1, 3, 1])', 'array([2, 1, 3, 1])', 'array([2, 2, 2, 3])', 'array([1, 3, 2, 1])', 'array([1, 3, 2, 0])', 'array([0, 0, 1, 2])', 'array([3, 3, 0, 1])', 'array([1, 0, 1, 0])']
    obj_ins._seeds = np.array([1791095845, 2135392491, 946286476, 1857819720, 491263, 550290313, 1298508491, 2143362693, 630311759, 1013994432])
    obj_ins_pred = BaggingClassifier(estimator = pred_input['self']['estimator'], n_estimators = pred_input['self']['n_estimators'], max_samples = pred_input['self']['max_samples'], max_features = pred_input['self']['max_features'], bootstrap = pred_input['self']['bootstrap'], bootstrap_features = pred_input['self']['bootstrap_features'], oob_score = pred_input['self']['oob_score'], warm_start = pred_input['self']['warm_start'], n_jobs = pred_input['self']['n_jobs'], random_state = pred_input['self']['random_state'], verbose = pred_input['self']['verbose'])
    obj_ins_pred.estimator_params = pred_input['self']['estimator_params']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._n_samples = pred_input['self']['_n_samples']
    obj_ins_pred.classes_ = pred_input['self']['classes_']
    obj_ins_pred.n_classes_ = pred_input['self']['n_classes_']
    obj_ins_pred.estimator_ = pred_input['self']['estimator_']
    obj_ins_pred._max_samples = pred_input['self']['_max_samples']
    obj_ins_pred._max_features = pred_input['self']['_max_features']
    obj_ins_pred._sample_weight = pred_input['self']['_sample_weight']
    obj_ins_pred.estimators_ = pred_input['self']['estimators_']
    obj_ins_pred.estimators_features_ = pred_input['self']['estimators_features_']
    obj_ins_pred._seeds = pred_input['self']['_seeds']
    assert obj_ins.predict_log_proba(X = np.array([[ 2.24968346e+00, -1.05276654e+00, 1.78583195e+00, 1.44883158e+00], [-5.25060772e-02, -8.22569778e-01, 7.62758269e-01, 9.22302838e-01], [ 3.10997534e-01, -1.31979479e-01, 6.49083415e-01, 7.90670654e-01], [-7.79513300e-01, 7.88807586e-01, -1.34022653e+00, -1.31544430e+00], [ 1.03800476e+00, 9.82172869e-02, 5.35408562e-01, 3.95774101e-01], [-1.74885626e+00, -1.31979479e-01, -1.39706395e+00, -1.31544430e+00], [ 1.15917263e+00, -1.31979479e-01, 9.90107977e-01, 1.18556721e+00], [ 6.74501145e-01, -5.92373012e-01, 1.04694540e+00, 1.18556721e+00], [ 6.86617933e-02, -1.31979479e-01, 7.62758269e-01, 7.90670654e-01], [-1.73673948e-01, -1.31979479e-01, 2.51221427e-01, 8.77547895e-04], [-1.02184904e+00, -1.31979479e-01, -1.22655167e+00, -1.31544430e+00], [ 2.24968346e+00, -1.31979479e-01, 1.33113254e+00, 1.44883158e+00], [-1.73673948e-01, 3.09077525e+00, -1.28338910e+00, -1.05217993e+00], [ 5.53333275e-01, -8.22569778e-01, 6.49083415e-01, 7.90670654e-01], [-2.94841818e-01, -8.22569778e-01, 2.51221427e-01, 1.32509732e-01], [ 1.89829664e-01, -8.22569778e-01, 7.62758269e-01, 5.27406285e-01], [-1.50652052e+00, 7.88807586e-01, -1.34022653e+00, -1.18381211e+00], [ 1.03800476e+00, 5.58610819e-01, 1.10378283e+00, 1.18556721e+00], [-1.02184904e+00, -2.43394714e+00, -1.46640561e-01, -2.62386821e-01], [ 1.89829664e-01, -3.62176246e-01, 4.21733708e-01, 3.95774101e-01], [ 2.24968346e+00, -5.92373012e-01, 1.67215710e+00, 1.05393502e+00], [ 3.10997534e-01, -1.31979479e-01, 4.78571135e-01, 2.64141916e-01], [ 4.32165405e-01, -5.92373012e-01, 5.92245988e-01, 7.90670654e-01], [ 1.15917263e+00, -5.92373012e-01, 5.92245988e-01, 2.64141916e-01], [ 5.53333275e-01, 7.88807586e-01, 1.04694540e+00, 1.58046376e+00], [-1.26418478e+00, 7.88807586e-01, -1.05603939e+00, -1.31544430e+00], [-5.37177559e-01, 1.47939788e+00, -1.28338910e+00, -1.31544430e+00], [-1.02184904e+00, 1.24920112e+00, -1.34022653e+00, -1.31544430e+00], [-1.14301691e+00, -1.31979479e-01, -1.34022653e+00, -1.31544430e+00], [-9.00681170e-01, 1.70959465e+00, -1.05603939e+00, -1.05217993e+00], [-5.25060772e-02, -8.22569778e-01, 7.62758269e-01, 9.22302838e-01], [ 1.28034050e+00, 3.28414053e-01, 1.10378283e+00, 1.44883158e+00], [ 4.32165405e-01, -3.62176246e-01, 3.08058854e-01, 1.32509732e-01], [-1.73673948e-01, -5.92373012e-01, 1.94384000e-01, 1.32509732e-01], [ 5.53333275e-01, -1.28296331e+00, 6.49083415e-01, 3.95774101e-01], [ 7.95669016e-01, -1.31979479e-01, 8.19595696e-01, 1.05393502e+00], [-1.26418478e+00, -1.31979479e-01, -1.34022653e+00, -1.18381211e+00], [ 1.52267624e+00, -1.31979479e-01, 1.21745768e+00, 1.18556721e+00]]))==obj_ins_pred.predict_log_proba(X = pred_input['args']['X']), 'Prediction failed!'