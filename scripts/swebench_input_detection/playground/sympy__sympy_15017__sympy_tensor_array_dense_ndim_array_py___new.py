# Problem: sympy__sympy-15017@@sympy.tensor.array.dense_ndim_array.py@@_new
# Benchmark: Swebench
# Module: sympy.tensor.array.dense_ndim_array
# Function: _new

from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray


def test_input(pred_input):
    obj_ins = ImmutableDenseNDimArray()
    obj_ins_pred = ImmutableDenseNDimArray()
    assert obj_ins._new(cls = {'__module__': 'sympy.tensor.array.dense_ndim_array', '__doc__': '\n\n    ', '__new__': {}, '_new': {}, '__setitem__': {}, 'as_mutable': {}, '_explicit_class_assumptions': {}, 'default_assumptions': {'_generator': {}}, '_prop_handler': {}}, iterable = {'rows': 2, 'cols': 2, '_mat': None}, shape = None)==obj_ins_pred._new(cls = pred_input['args']['cls'], iterable = pred_input['args']['iterable'], shape = pred_input['args']['shape']), 'Prediction failed!'