# Problem: sympy@@sympy_printing_pretty_stringpict.py@@render_L249
# Module: sympy.printing.pretty.stringpict
# Function: render
# Line: 249

from sympy.printing.pretty.stringpict import stringPict


def test_input(pred_input):
    obj_ins = stringPict(s = '{Q.integer, Q.positive}', baseline = 0)
    obj_ins.picture = ['{Q.integer, Q.positive}']
    obj_ins.binding = 0
    obj_ins._unicode = '{Q.integer, Q.positive}'
    obj_ins_pred = stringPict(s = pred_input['self']['s'], baseline = pred_input['self']['baseline'])
    obj_ins_pred.picture = pred_input['self']['picture']
    obj_ins_pred.binding = pred_input['self']['binding']
    obj_ins_pred._unicode = pred_input['self']['_unicode']
    assert obj_ins.render(args = [], order = None, full_prec = 'auto', use_unicode = None, wrap_line = True, num_columns = None, use_unicode_sqrt_char = True, root_notation = True, mat_symbol_style = 'plain', imaginary_unit = 'i', perm_cyclic = True)==obj_ins_pred.render(args = pred_input['args']['args'], order = pred_input['kwargs']['order'], full_prec = pred_input['kwargs']['full_prec'], use_unicode = pred_input['kwargs']['use_unicode'], wrap_line = pred_input['kwargs']['wrap_line'], num_columns = pred_input['kwargs']['num_columns'], use_unicode_sqrt_char = pred_input['kwargs']['use_unicode_sqrt_char'], root_notation = pred_input['kwargs']['root_notation'], mat_symbol_style = pred_input['kwargs']['mat_symbol_style'], imaginary_unit = pred_input['kwargs']['imaginary_unit'], perm_cyclic = pred_input['kwargs']['perm_cyclic']), 'Prediction failed!'
    
