# Problem: sympy@@sympy_printing_printer.py@@_print_L295
# Module: sympy.printing.printer
# Function: _print
# Line: 295

from sympy.printing.printer import Printer


def test_input(pred_input):
    obj_ins = Printer()
    obj_ins._str = "<class 'str'>"
    obj_ins._settings = {'order': None, 'full_prec': 'auto', 'sympy_integers': False, 'abbrev': False, 'perm_cyclic': True, 'min': None, 'max': None, 'dps': None}
    obj_ins._context = {}
    obj_ins._print_level = 1
    obj_ins_pred = Printer()
    obj_ins_pred._str = pred_input['self']['_str']
    obj_ins_pred._settings = pred_input['self']['_settings']
    obj_ins_pred._context = pred_input['self']['_context']
    obj_ins_pred._print_level = pred_input['self']['_print_level']
    assert obj_ins._print(expr = 'phi')==obj_ins_pred._print(expr = pred_input['args']['expr']), 'Prediction failed!'
    
