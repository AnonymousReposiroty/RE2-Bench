# Problem: sphinx-doc__sphinx-10551@@sphinx.pycode.ast.py@@visit_UnaryOp
# Benchmark: Swebench
# Module: sphinxcode.ast
# Function: visit_UnaryOp

from sphinxcode.ast import _UnparseVisitor


def test_input(pred_input):
    obj_ins = _UnparseVisitor()
    obj_ins.code = '~1'
    obj_ins_pred = _UnparseVisitor()
    obj_ins_pred.code = pred_input['self']['code']
    assert  {'op': {}, 'operand': {'value': 1, 'kind': None, 'lineno': 1, 'col_offset': 1, 'end_lineno': 1, 'end_col_offset': 2}, 'lineno': 1, 'col_offset': 0, 'end_lineno': 1, 'end_col_offset': 2} == pred_input['args']['node'], 'Prediction failed!'
    
