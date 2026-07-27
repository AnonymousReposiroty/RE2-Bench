from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .symbol import Dummy

def _mask_nc(eq, name=None):
    """
    Return ``eq`` with non-commutative objects replaced with Dummy
    symbols. A dictionary that can be used to restore the original
    values is returned: if it is None, the expression is noncommutative
    and cannot be made commutative. The third value returned is a list
    of any non-commutative symbols that appear in the returned equation.

    Explanation
    ===========

    All non-commutative objects other than Symbols are replaced with
    a non-commutative Symbol. Identical objects will be identified
    by identical symbols.

    If there is only 1 non-commutative object in an expression it will
    be replaced with a commutative symbol. Otherwise, the non-commutative
    entities are retained and the calling routine should handle
    replacements in this case since some care must be taken to keep
    track of the ordering of symbols when they occur within Muls.

    Parameters
    ==========

    name : str
        ``name``, if given, is the name that will be used with numbered Dummy
        variables that will replace the non-commutative objects and is mainly
        used for doctesting purposes.

    Examples
    ========

    >>> from sympy.physics.secondquant import Commutator, NO, F, Fd
    >>> from sympy import symbols
    >>> from sympy.core.exprtools import _mask_nc
    >>> from sympy.abc import x, y
    >>> A, B, C = symbols('A,B,C', commutative=False)

    One nc-symbol:

    >>> _mask_nc(A**2 - x**2, 'd')
    (_d0**2 - x**2, {_d0: A}, [])

    Multiple nc-symbols:

    >>> _mask_nc(A**2 - B**2, 'd')
    (A**2 - B**2, {}, [A, B])

    An nc-object with nc-symbols but no others outside of it:

    >>> _mask_nc(1 + x*Commutator(A, B), 'd')
    (_d0*x + 1, {_d0: Commutator(A, B)}, [])
    >>> _mask_nc(NO(Fd(x)*F(y)), 'd')
    (_d0, {_d0: NO(CreateFermion(x)*AnnihilateFermion(y))}, [])

    Multiple nc-objects:

    >>> eq = x*Commutator(A, B) + x*Commutator(A, C)*Commutator(A, B)
    >>> _mask_nc(eq, 'd')
    (x*_d0 + x*_d1*_d0, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1])

    Multiple nc-objects and nc-symbols:

    >>> eq = A*Commutator(A, B) + B*Commutator(A, C)
    >>> _mask_nc(eq, 'd')
    (A*_d0 + B*_d1, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1, A, B])

    """
    name = name or 'mask'
    # Make Dummy() append sequential numbers to the name

    def numbered_names():
        i = 0
        while True:
            yield name + str(i)
            i += 1

    names = numbered_names()

    def Dummy(*args, **kwargs):
        from .symbol import Dummy
        return Dummy(next(names), *args, **kwargs)

    expr = eq
    if expr.is_commutative:
        return eq, {}, []

    # identify nc-objects; symbols and other
    rep = []
    nc_obj = set()
    nc_syms = set()
    pot = preorder_traversal(expr, keys=default_sort_key)
    for a in pot:
        if any(a == r[0] for r in rep):
            pot.skip()
        elif not a.is_commutative:
            if a.is_symbol:
                nc_syms.add(a)
                pot.skip()
            elif not (a.is_Add or a.is_Mul or a.is_Pow):
                nc_obj.add(a)
                pot.skip()

    # If there is only one nc symbol or object, it can be factored regularly
    # but polys is going to complain, so replace it with a Dummy.
    if len(nc_obj) == 1 and not nc_syms:
        rep.append((nc_obj.pop(), Dummy()))
    elif len(nc_syms) == 1 and not nc_obj:
        rep.append((nc_syms.pop(), Dummy()))

    # Any remaining nc-objects will be replaced with an nc-Dummy and
    # identified as an nc-Symbol to watch out for
    nc_obj = sorted(nc_obj, key=default_sort_key)
    for n in nc_obj:
        nc = Dummy(commutative=False)
        rep.append((n, nc))
        nc_syms.add(nc)
    expr = expr.subs(rep)

    nc_syms = list(nc_syms)
    nc_syms.sort(key=default_sort_key)
    return expr, {v: k for k, v in rep}, nc_syms
