# Problem: sympy@@sympy_core_exprtools.py@@mul_L857
# Module: sympy.core.exprtools
# Function: mul
# Line: 857

from sympy.core.exprtools import Term, Factors
from sympy.core.expr import Expr
from sympy import sympify

def recover_term(term_str: str) -> Term:
    # Environment in which the string will be interpreted
    local_dict = {
        "Term": Term,
        "Factors": Factors,
    }

    # sympify will turn bare names (x, y, etc.) into Symbols automatically
    term = sympify(term_str, locals=local_dict)

    if not isinstance(term, Term):
        raise TypeError(f"String did not evaluate to a Term, got {type(term)} instead")

    return term

def test_input(pred_input):
    obj_ins = recover_term("Term(1, Factors({}), Factors({}))")
    obj_ins_pred = recover_term(pred_input['self']["__repr__"])
    assert obj_ins.mul(other = recover_term('Term(1, Factors({}), Factors({}))'))==obj_ins_pred.mul(other = recover_term(pred_input['args']['other'])), 'Prediction failed!'
