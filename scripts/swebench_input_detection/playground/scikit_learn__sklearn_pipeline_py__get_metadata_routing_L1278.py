import numpy as np

# Problem: scikit-learn@@sklearn_pipeline.py@@get_metadata_routing_L1278
# Module: sklearn.pipeline
# Function: get_metadata_routing
# Line: 1278

from sklearn.pipeline import Pipeline


def test_input(pred_input):
    obj_ins = Pipeline(steps = [['consumesmetadata', 'ConsumesMetadata(on_fit=True, on_predict=True)']], transform_input = None, memory = None, verbose = False)
    obj_ins_pred = Pipeline(steps = pred_input['self']['steps'], transform_input = pred_input['self']['transform_input'], memory = pred_input['self']['memory'], verbose = pred_input['self']['verbose'])
    assert obj_ins.get_metadata_routing()==obj_ins_pred.get_metadata_routing(), 'Prediction failed!'

