# Problem: sympy@@sympy_printing_str.py@@_print_Pow_L608
# Module: sympy.printing.str
# Function: _print_Pow
# Line: 608

from sympy.printing.str import StrPrinter
from sympy import sympify

def test_input(pred_input):
    obj_ins = StrPrinter()
    obj_ins._str = "<class 'str'>"
    obj_ins._settings = {'order': None, 'full_prec': 'auto', 'sympy_integers': False, 'abbrev': False, 'perm_cyclic': True, 'min': None, 'max': None, 'dps': None}
    obj_ins._context = {}
    obj_ins._print_level = 1
    obj_ins_pred = StrPrinter()
    obj_ins_pred._str = pred_input['self']['_str']
    obj_ins_pred._settings = pred_input['self']['_settings']
    obj_ins_pred._context = pred_input['self']['_context']
    obj_ins_pred._print_level = pred_input['self']['_print_level']
    assert obj_ins._print_Pow(expr = sympify('sqrt(2)'), rational = False)==obj_ins_pred._print_Pow(expr = sympify(pred_input['args']['expr']), rational = pred_input['args']['rational']), 'Prediction failed!'
    
    