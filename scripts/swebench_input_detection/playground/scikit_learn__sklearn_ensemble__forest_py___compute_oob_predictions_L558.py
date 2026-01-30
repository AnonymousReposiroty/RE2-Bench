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

# Problem: scikit-learn@@sklearn_ensemble__forest.py@@_compute_oob_predictions_L558
# Module: sklearn.ensemble._forest
# Function: _compute_oob_predictions
# Line: 558

from sklearn.ensemble._forest import BaseForest


def test_input(pred_input):
    obj_ins = BaseForest(estimator = 'DecisionTreeClassifier()', n_estimators = 40, estimator_params = ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'random_state', 'ccp_alpha', 'monotonic_cst'], bootstrap = True, oob_score = True, n_jobs = None, random_state = 0, verbose = 0, warm_start = False, class_weight = None, max_samples = None)
    obj_ins.criterion = 'gini'
    obj_ins.max_depth = None
    obj_ins.min_samples_split = 2
    obj_ins.min_samples_leaf = 1
    obj_ins.min_weight_fraction_leaf = 0.0
    obj_ins.max_features = 'sqrt'
    obj_ins.max_leaf_nodes = None
    obj_ins.min_impurity_decrease = 0.0
    obj_ins.monotonic_cst = None
    obj_ins.ccp_alpha = 0.0
    obj_ins.n_features_in_ = 20
    obj_ins._n_samples = 150
    obj_ins.n_outputs_ = 1
    obj_ins.classes_ = [np.array([0, 1])]
    obj_ins.n_classes_ = [2]
    obj_ins._n_samples_bootstrap = 150
    obj_ins.estimator_ = 'DecisionTreeClassifier()'
    obj_ins.estimators_ = ["DecisionTreeClassifier(max_features='sqrt', random_state=209652396)", "DecisionTreeClassifier(max_features='sqrt', random_state=398764591)", "DecisionTreeClassifier(max_features='sqrt', random_state=924231285)", "DecisionTreeClassifier(max_features='sqrt', random_state=1478610112)", "DecisionTreeClassifier(max_features='sqrt', random_state=441365315)", "DecisionTreeClassifier(max_features='sqrt', random_state=1537364731)", "DecisionTreeClassifier(max_features='sqrt', random_state=192771779)", "DecisionTreeClassifier(max_features='sqrt', random_state=1491434855)", "DecisionTreeClassifier(max_features='sqrt', random_state=1819583497)", "DecisionTreeClassifier(max_features='sqrt', random_state=530702035)", "DecisionTreeClassifier(max_features='sqrt', random_state=626610453)", "DecisionTreeClassifier(max_features='sqrt', random_state=1650906866)", "DecisionTreeClassifier(max_features='sqrt', random_state=1879422756)", "DecisionTreeClassifier(max_features='sqrt', random_state=1277901399)", "DecisionTreeClassifier(max_features='sqrt', random_state=1682652230)", "DecisionTreeClassifier(max_features='sqrt', random_state=243580376)", "DecisionTreeClassifier(max_features='sqrt', random_state=1991416408)", "DecisionTreeClassifier(max_features='sqrt', random_state=1171049868)", "DecisionTreeClassifier(max_features='sqrt', random_state=1646868794)", "DecisionTreeClassifier(max_features='sqrt', random_state=2051556033)", "DecisionTreeClassifier(max_features='sqrt', random_state=1252949478)", "DecisionTreeClassifier(max_features='sqrt', random_state=1340754471)", "DecisionTreeClassifier(max_features='sqrt', random_state=124102743)", "DecisionTreeClassifier(max_features='sqrt', random_state=2061486254)", "DecisionTreeClassifier(max_features='sqrt', random_state=292249176)", "DecisionTreeClassifier(max_features='sqrt', random_state=1686997841)", "DecisionTreeClassifier(max_features='sqrt', random_state=1827923621)", "DecisionTreeClassifier(max_features='sqrt', random_state=1443447321)", "DecisionTreeClassifier(max_features='sqrt', random_state=305097549)", "DecisionTreeClassifier(max_features='sqrt', random_state=1449105480)", "DecisionTreeClassifier(max_features='sqrt', random_state=374217481)", "DecisionTreeClassifier(max_features='sqrt', random_state=636393364)", "DecisionTreeClassifier(max_features='sqrt', random_state=86837363)", "DecisionTreeClassifier(max_features='sqrt', random_state=1581585360)", "DecisionTreeClassifier(max_features='sqrt', random_state=1428591347)", "DecisionTreeClassifier(max_features='sqrt', random_state=1963466437)", "DecisionTreeClassifier(max_features='sqrt', random_state=1194674174)", "DecisionTreeClassifier(max_features='sqrt', random_state=602801999)", "DecisionTreeClassifier(max_features='sqrt', random_state=1589190063)", "DecisionTreeClassifier(max_features='sqrt', random_state=1589512640)"]
    obj_ins_pred = BaseForest(estimator = pred_input['self']['estimator'], n_estimators = pred_input['self']['n_estimators'], estimator_params = pred_input['self']['estimator_params'], bootstrap = pred_input['self']['bootstrap'], oob_score = pred_input['self']['oob_score'], n_jobs = pred_input['self']['n_jobs'], random_state = pred_input['self']['random_state'], verbose = pred_input['self']['verbose'], warm_start = pred_input['self']['warm_start'], class_weight = pred_input['self']['class_weight'], max_samples = pred_input['self']['max_samples'])
    obj_ins_pred.criterion = pred_input['self']['criterion']
    obj_ins_pred.max_depth = pred_input['self']['max_depth']
    obj_ins_pred.min_samples_split = pred_input['self']['min_samples_split']
    obj_ins_pred.min_samples_leaf = pred_input['self']['min_samples_leaf']
    obj_ins_pred.min_weight_fraction_leaf = pred_input['self']['min_weight_fraction_leaf']
    obj_ins_pred.max_features = pred_input['self']['max_features']
    obj_ins_pred.max_leaf_nodes = pred_input['self']['max_leaf_nodes']
    obj_ins_pred.min_impurity_decrease = pred_input['self']['min_impurity_decrease']
    obj_ins_pred.monotonic_cst = pred_input['self']['monotonic_cst']
    obj_ins_pred.ccp_alpha = pred_input['self']['ccp_alpha']
    obj_ins_pred.n_features_in_ = pred_input['self']['n_features_in_']
    obj_ins_pred._n_samples = pred_input['self']['_n_samples']
    obj_ins_pred.n_outputs_ = pred_input['self']['n_outputs_']
    obj_ins_pred.classes_ = pred_input['self']['classes_']
    obj_ins_pred.n_classes_ = pred_input['self']['n_classes_']
    obj_ins_pred._n_samples_bootstrap = pred_input['self']['_n_samples_bootstrap']
    obj_ins_pred.estimator_ = pred_input['self']['estimator_']
    obj_ins_pred.estimators_ = pred_input['self']['estimators_']
    assert obj_ins._compute_oob_predictions(X = [], y = np.array([[0.], [0.], [1.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [1.], [0.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [1.], [1.], [1.], [0.], [0.], [1.], [0.], [0.], [1.], [1.], [1.], [1.], [1.], [1.], [0.], [1.], [0.], [1.], [0.], [1.], [1.], [1.], [1.], [1.], [1.], [0.], [1.], [1.], [1.], [0.], [1.], [0.], [1.], [0.], [0.], [1.], [0.], [1.], [1.], [1.], [0.], [1.], [1.], [0.], [1.], [1.], [1.], [0.], [1.], [1.], [1.], [0.], [1.], [1.], [1.], [0.], [0.], [1.], [1.], [1.], [1.], [1.], [0.], [1.], [1.], [0.], [1.], [0.], [1.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [1.], [1.], [1.], [0.], [1.], [0.], [0.], [0.], [1.], [0.], [1.], [1.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [1.], [1.], [1.], [0.], [0.], [1.], [1.], [0.], [1.], [1.], [0.], [0.], [1.], [1.], [1.], [0.], [0.], [1.], [0.], [0.], [1.], [0.]]))==obj_ins_pred._compute_oob_predictions(X = pred_input['args']['X'], y = pred_input['args']['y']), 'Prediction failed!'
