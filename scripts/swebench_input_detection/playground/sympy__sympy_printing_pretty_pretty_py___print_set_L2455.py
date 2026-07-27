# Problem: sympy@@sympy_printing_pretty_pretty.py@@_print_set_L2455
# Module: sympy.printing.pretty.pretty
# Function: _print_set
# Line: 2455

from sympy.printing.pretty.pretty import PrettyPrinter


def test_input(pred_input):
    obj_ins = PrettyPrinter()
    obj_ins._str = "<class 'str'>"
    obj_ins._settings = {'order': None, 'full_prec': 'auto', 'use_unicode': None, 'wrap_line': True, 'num_columns': None, 'use_unicode_sqrt_char': True, 'root_notation': True, 'mat_symbol_style': 'plain', 'imaginary_unit': 'i', 'perm_cyclic': True}
    obj_ins._context = {}
    obj_ins._print_level = 1
    obj_ins_pred = PrettyPrinter()
    obj_ins_pred._str = pred_input['self']['_str']
    obj_ins_pred._settings = pred_input['self']['_settings']
    obj_ins_pred._context = pred_input['self']['_context']
    obj_ins_pred._print_level = pred_input['self']['_print_level']
    assert obj_ins._print_set(s = ['Q.integer', 'Q.positive'])==obj_ins_pred._print_set(s = pred_input['args']['s']), 'Prediction failed!'
    