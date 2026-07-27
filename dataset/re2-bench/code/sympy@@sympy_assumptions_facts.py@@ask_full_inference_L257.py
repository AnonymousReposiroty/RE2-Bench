from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,
    Exclusive,)
from sympy.logic.inference import satisfiable

def ask_full_inference(proposition, assumptions, known_facts_cnf):
    """
    Method for inferring properties about objects.

    """
    if not satisfiable(And(known_facts_cnf, assumptions, proposition)):
        return False
    if not satisfiable(And(known_facts_cnf, assumptions, Not(proposition))):
        return True
    return None
