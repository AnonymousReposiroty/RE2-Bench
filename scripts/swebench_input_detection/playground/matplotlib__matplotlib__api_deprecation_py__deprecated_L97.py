# Problem: matplotlib@@matplotlib__api_deprecation.py@@deprecated_L97
# Module: matplotlib._api.deprecation
# Function: deprecated
# Line: 97

from matplotlib._api.deprecation import deprecated


def test_input(pred_input):
    assert deprecated(since = '0.0.0', message = '', name = '', alternative = '', pending = False, obj_type = None, addendum = '', removal = '')==deprecated(since = pred_input['args']['since'], message = pred_input['args']['message'], name = pred_input['args']['name'], alternative = pred_input['args']['alternative'], pending = pred_input['args']['pending'], obj_type = pred_input['args']['obj_type'], addendum = pred_input['args']['addendum'], removal = pred_input['args']['removal']), 'Prediction failed!'
    
    
    