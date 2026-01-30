from sympy.core.basic import Basic, as_Basic
from sympy.core.sorting import ordered
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
    true, false, Or, ITE, simplify_logic, to_cnf, distribute_or_over_and)
from sympy.utilities.iterables import uniq, sift, common_prefix
from itertools import product

def piecewise_fold(expr, evaluate=True):
    """
    Takes an expression containing a piecewise function and returns the
    expression in piecewise form. In addition, any ITE conditions are
    rewritten in negation normal form and simplified.

    The final Piecewise is evaluated (default) but if the raw form
    is desired, send ``evaluate=False``; if trivial evaluation is
    desired, send ``evaluate=None`` and duplicate conditions and
    processing of True and False will be handled.

    Examples
    ========

    >>> from sympy import Piecewise, piecewise_fold, S
    >>> from sympy.abc import x
    >>> p = Piecewise((x, x < 1), (1, S(1) <= x))
    >>> piecewise_fold(x*p)
    Piecewise((x**2, x < 1), (x, True))

    See Also
    ========

    Piecewise
    piecewise_exclusive
    """
    if not isinstance(expr, Basic) or not expr.has(Piecewise):
        return expr

    new_args = []
    if isinstance(expr, (ExprCondPair, Piecewise)):
        for e, c in expr.args:
            if not isinstance(e, Piecewise):
                e = piecewise_fold(e)
            # we don't keep Piecewise in condition because
            # it has to be checked to see that it's complete
            # and we convert it to ITE at that time
            assert not c.has(Piecewise)  # pragma: no cover
            if isinstance(c, ITE):
                c = c.to_nnf()
                c = simplify_logic(c, form='cnf')
            if isinstance(e, Piecewise):
                new_args.extend([(piecewise_fold(ei), And(ci, c))
                    for ei, ci in e.args])
            else:
                new_args.append((e, c))
    else:
        # Given
        #     P1 = Piecewise((e11, c1), (e12, c2), A)
        #     P2 = Piecewise((e21, c1), (e22, c2), B)
        #     ...
        # the folding of f(P1, P2) is trivially
        # Piecewise(
        #   (f(e11, e21), c1),
        #   (f(e12, e22), c2),
        #   (f(Piecewise(A), Piecewise(B)), True))
        # Certain objects end up rewriting themselves as thus, so
        # we do that grouping before the more generic folding.
        # The following applies this idea when f = Add or f = Mul
        # (and the expression is commutative).
        if expr.is_Add or expr.is_Mul and expr.is_commutative:
            p, args = sift(expr.args, lambda x: x.is_Piecewise, binary=True)
            pc = sift(p, lambda x: tuple([c for e,c in x.args]))
            for c in list(ordered(pc)):
                if len(pc[c]) > 1:
                    pargs = [list(i.args) for i in pc[c]]
                    # the first one is the same; there may be more
                    com = common_prefix(*[
                        [i.cond for i in j] for j in pargs])
                    n = len(com)
                    collected = []
                    for i in range(n):
                        collected.append((
                            expr.func(*[ai[i].expr for ai in pargs]),
                            com[i]))
                    remains = []
                    for a in pargs:
                        if n == len(a):  # no more args
                            continue
                        if a[n].cond == True:  # no longer Piecewise
                            remains.append(a[n].expr)
                        else:  # restore the remaining Piecewise
                            remains.append(
                                Piecewise(*a[n:], evaluate=False))
                    if remains:
                        collected.append((expr.func(*remains), True))
                    args.append(Piecewise(*collected, evaluate=False))
                    continue
                args.extend(pc[c])
        else:
            args = expr.args
        # fold
        folded = list(map(piecewise_fold, args))
        for ec in product(*[
                (i.args if isinstance(i, Piecewise) else
                 [(i, true)]) for i in folded]):
            e, c = zip(*ec)
            new_args.append((expr.func(*e), And(*c)))

    if evaluate is None:
        # don't return duplicate conditions, otherwise don't evaluate
        new_args = list(reversed([(e, c) for c, e in {
            c: e for e, c in reversed(new_args)}.items()]))
    rv = Piecewise(*new_args, evaluate=evaluate)
    if evaluate is None and len(rv.args) == 1 and rv.args[0].cond == True:
        return rv.args[0].expr
    if any(s.expr.has(Piecewise) for p in rv.atoms(Piecewise) for s in p.args):
        return piecewise_fold(rv)
    return rv
