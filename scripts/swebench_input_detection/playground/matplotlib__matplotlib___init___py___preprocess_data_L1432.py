# Problem: matplotlib@@matplotlib___init__.py@@_preprocess_data_L1432
# Module: matplotlib
# Function: _preprocess_data
# Line: 1432

from matplotlib import _preprocess_data

def func(ax, x, y):
    # do something with x, y
    return x, y

def test_input(pred_input):
    assert _preprocess_data(func = func, replace_names = ['x', 'y'], label_namer = None)==_preprocess_data(func = pred_input['args']['func'], replace_names = pred_input['args']['replace_names'], label_namer = pred_input['args']['label_namer']), 'Prediction failed!'
    
print(_preprocess_data(func = func, replace_names = ['x', 'y'], label_namer = None))
