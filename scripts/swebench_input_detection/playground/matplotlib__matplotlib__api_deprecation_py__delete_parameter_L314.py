# Problem: matplotlib@@matplotlib__api_deprecation.py@@delete_parameter_L314
# Module: matplotlib._api.deprecation
# Function: delete_parameter
# Line: 314

from matplotlib._api.deprecation import delete_parameter

def testfunc(foo):
    pass
def test_input(pred_input):
    assert delete_parameter(since = '3.0', name = 'foo', func = testfunc)==delete_parameter(since = pred_input['args']['since'], name = pred_input['args']['name'], func = testfunc), 'Prediction failed!'
