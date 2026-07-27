# Problem: sympy__sympy-21055@@sympy.assumptions.refine.py@@refine_arg
# Benchmark: Swebench
# Module: sympy.assumptions.refine
# Function: refine_arg

from sympy.assumptions.refine import refine_arg


def test_input(pred_input):
    assert refine_arg(expr = {'nargs': {'_args_set': "{'1'}"}}, assumptions = 'Q.positive(x)')==refine_arg(expr = pred_input['args']['expr'], assumptions = pred_input['args']['assumptions']), 'Prediction failed!'
    
