import numpy as np

# Problem: scikit-learn__scikit-learn-11315@@sklearn.utils.metaestimators.py@@_get_params
# Benchmark: Swebench
# Module: sklearn.utils.metaestimators
# Function: _get_params

from sklearn.utils.metaestimators import _BaseComposition
class DummyComposition(_BaseComposition):
    def __init__(self):
        self.transformers = []
        self.remainder = 'passthrough'
        self.n_jobs = 1
        self.transformer_weights = None
        self.transformers_ = self.transformers
        self._remainder = ('remainder', 'passthrough', None)    

def test_input(pred_input):
    obj_ins = DummyComposition()
    obj_ins.transformers = None
    obj_ins.remainder = 'passthrough'
    obj_ins.n_jobs = 1
    obj_ins.transformer_weights = None
    obj_ins._remainder = ['remainder', 'passthrough', None]
    obj_ins.transformers_ = None
    obj_ins_pred = DummyComposition()
    obj_ins_pred.transformers = pred_input['self']['transformers']
    obj_ins_pred.remainder = pred_input['self']['remainder']
    obj_ins_pred.n_jobs = pred_input['self']['n_jobs']
    obj_ins_pred.transformer_weights = pred_input['self']['transformer_weights']
    obj_ins_pred._remainder = pred_input['self']['_remainder']
    obj_ins_pred.transformers_ = pred_input['self']['transformers_']
    assert obj_ins._get_params(attr = '_transformers', deep = False)==obj_ins_pred._get_params(attr = pred_input['args']['attr'], deep = pred_input['kwargs']['deep']), 'Prediction failed!'
