import numpy as np


from sklearn.preprocessing._label import MultiLabelBinarizer


def test_input(pred_input):
    obj_ins = MultiLabelBinarizer(classes = None, sparse_output = False)
    obj_ins._cached_dict = None
    obj_ins.classes_ = np.array([0, 1, 2])
    obj_ins_pred = MultiLabelBinarizer(classes = pred_input['self']['classes'], sparse_output = pred_input['self']['sparse_output'])
    obj_ins_pred._cached_dict = pred_input['self']['_cached_dict']
    obj_ins_pred.classes_ = pred_input['self']['classes_']
    assert obj_ins.transform(y = [[np.int64(1)], [np.int64(1)], [np.int64(0), np.int64(1), np.int64(2)], [np.int64(0), np.int64(1), np.int64(2)], [np.int64(1)], [np.int64(0), np.int64(1)], [], [np.int64(0)], [np.int64(2)], [], [np.int64(1)], [], [np.int64(1), np.int64(2)], [np.int64(0)], [np.int64(0), np.int64(1)], [np.int64(0), np.int64(1), np.int64(2)], [], [np.int64(1)], [], [np.int64(0), np.int64(1), np.int64(2)], [np.int64(1), np.int64(2)], [np.int64(2)], [np.int64(0), np.int64(2)], [np.int64(1)], [np.int64(1)]])==obj_ins_pred.transform(y = pred_input['args']['y']), 'Prediction failed!'