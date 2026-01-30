from sympy.assumptions.ask import Q
from sympy.core.relational import Eq, Ne

def _eval_binrel(binrel):
    """
    Simplify binary relation to True / False if possible.
    """
    if not (len(binrel.lhs.free_symbols) == 0 and len(binrel.rhs.free_symbols) == 0):
        return binrel
    if binrel.function == Q.lt:
        res = binrel.lhs < binrel.rhs
    elif binrel.function == Q.gt:
        res = binrel.lhs > binrel.rhs
    elif binrel.function == Q.le:
        res = binrel.lhs <= binrel.rhs
    elif binrel.function == Q.ge:
        res = binrel.lhs >= binrel.rhs
    elif binrel.function == Q.eq:
        res = Eq(binrel.lhs, binrel.rhs)
    elif binrel.function == Q.ne:
        res = Ne(binrel.lhs, binrel.rhs)

    if res == True or res == False:
        return res
    else:
        return None
