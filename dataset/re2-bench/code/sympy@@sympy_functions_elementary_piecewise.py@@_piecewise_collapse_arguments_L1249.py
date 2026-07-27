from sympy.core import S, diff, Tuple, Dummy, Mul
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
    _canonical, _canonical_coeff)
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
    true, false, Or, ITE, simplify_logic, to_cnf, distribute_or_over_and)

def _piecewise_collapse_arguments(_args):
    newargs = []  # the unevaluated conditions
    current_cond = set()  # the conditions up to a given e, c pair
    for expr, cond in _args:
        cond = cond.replace(
            lambda _: _.is_Relational, _canonical_coeff)
        # Check here if expr is a Piecewise and collapse if one of
        # the conds in expr matches cond. This allows the collapsing
        # of Piecewise((Piecewise((x,x<0)),x<0)) to Piecewise((x,x<0)).
        # This is important when using piecewise_fold to simplify
        # multiple Piecewise instances having the same conds.
        # Eventually, this code should be able to collapse Piecewise's
        # having different intervals, but this will probably require
        # using the new assumptions.
        if isinstance(expr, Piecewise):
            unmatching = []
            for i, (e, c) in enumerate(expr.args):
                if c in current_cond:
                    # this would already have triggered
                    continue
                if c == cond:
                    if c != True:
                        # nothing past this condition will ever
                        # trigger and only those args before this
                        # that didn't match a previous condition
                        # could possibly trigger
                        if unmatching:
                            expr = Piecewise(*(
                                unmatching + [(e, c)]))
                        else:
                            expr = e
                    break
                else:
                    unmatching.append((e, c))

        # check for condition repeats
        got = False
        # -- if an And contains a condition that was
        #    already encountered, then the And will be
        #    False: if the previous condition was False
        #    then the And will be False and if the previous
        #    condition is True then then we wouldn't get to
        #    this point. In either case, we can skip this condition.
        for i in ([cond] +
                  (list(cond.args) if isinstance(cond, And) else
                  [])):
            if i in current_cond:
                got = True
                break
        if got:
            continue

        # -- if not(c) is already in current_cond then c is
        #    a redundant condition in an And. This does not
        #    apply to Or, however: (e1, c), (e2, Or(~c, d))
        #    is not (e1, c), (e2, d) because if c and d are
        #    both False this would give no results when the
        #    true answer should be (e2, True)
        if isinstance(cond, And):
            nonredundant = []
            for c in cond.args:
                if isinstance(c, Relational):
                    if c.negated.canonical in current_cond:
                        continue
                    # if a strict inequality appears after
                    # a non-strict one, then the condition is
                    # redundant
                    if isinstance(c, (Lt, Gt)) and (
                        c.weak in current_cond):
                        cond = False
                        break
                nonredundant.append(c)
            else:
                cond = cond.func(*nonredundant)
        elif isinstance(cond, Relational):
            if cond.negated.canonical in current_cond:
                cond = S.true

        current_cond.add(cond)

        # collect successive e,c pairs when exprs or cond match
        if newargs:
            if newargs[-1].expr == expr:
                orcond = Or(cond, newargs[-1].cond)
                if isinstance(orcond, (And, Or)):
                    orcond = distribute_and_over_or(orcond)
                newargs[-1] = ExprCondPair(expr, orcond)
                continue
            elif newargs[-1].cond == cond:
                continue
        newargs.append(ExprCondPair(expr, cond))
    return newargs
