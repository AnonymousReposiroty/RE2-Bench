# Problem: sphinx-doc__sphinx-10551@@sphinx.pycode.ast.py@@visit_BinOp
# Benchmark: Swebench
# Module: sphinxcode.ast
# Function: visit_BinOp

from sphinxcode.ast import _UnparseVisitor


def test_input(pred_input):
    obj_ins = _UnparseVisitor()
    obj_ins.code = 'def "func(age: int | None)": pass'
    obj_ins_pred = _UnparseVisitor()
    obj_ins_pred.code = pred_input['self']['code']
    assert obj_ins.visit_BinOp(node = {'left': {'id': 'int', 'ctx': {}, 'lineno': 1, 'col_offset': 14, 'end_lineno': 1, 'end_col_offset': 17}, 'op': {}, 'right': {'value': None, 'kind': None, 'lineno': 1, 'col_offset': 20, 'end_lineno': 1, 'end_col_offset': 24}, 'lineno': 1, 'col_offset': 14, 'end_lineno': 1, 'end_col_offset': 24})==obj_ins_pred.visit_BinOp(node = pred_input['args']['node']), 'Prediction failed!'