# Problem: sympy@@sympy_core_function.py@@expand_log_L2905
# Module: sympy.core.function
# Function: expand_log
# Line: 2905

from sympy.core.function import expand_log
from sympy import sympify

def test_input(pred_input):
    assert expand_log(expr = sympify('log(2)'), deep = True, force = False, factor = False)==expand_log(expr = sympify(pred_input['args']['expr']), deep = pred_input['args']['deep'], force = pred_input['args']['force'], factor = pred_input['args']['factor']), 'Prediction failed!'
    
