def _linear_eq_to_dict(eqs, syms):
    """Convert a system Expr/Eq equations into dict form, returning
    the coefficient dictionaries and a list of syms-independent terms
    from each expression in ``eqs```.

    Examples
    ========

    >>> from sympy.polys.matrices.linsolve import _linear_eq_to_dict
    >>> from sympy.abc import x
    >>> _linear_eq_to_dict([2*x + 3], {x})
    ([{x: 2}], [3])
    """
    coeffs = []
    ind = []
    symset = set(syms)
    for e in eqs:
        if e.is_Equality:
            coeff, terms = _lin_eq2dict(e.lhs, symset)
            cR, tR = _lin_eq2dict(e.rhs, symset)
            # there were no nonlinear errors so now
            # cancellation is allowed
            coeff -= cR
            for k, v in tR.items():
                if k in terms:
                    terms[k] -= v
                else:
                    terms[k] = -v
            # don't store coefficients of 0, however
            terms = {k: v for k, v in terms.items() if v}
            c, d = coeff, terms
        else:
            c, d = _lin_eq2dict(e, symset)
        coeffs.append(d)
        ind.append(c)
    return coeffs, ind
