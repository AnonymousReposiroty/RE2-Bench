from sympy.core.basic import Basic
from sympy.core.sympify import sympify, _sympify
from sympy.polys.constructor import construct_domain
from sympy.polys.polyerrors import (
    OperationNotSupported, DomainError,
    CoercionFailed, UnificationFailed,
    GeneratorsNeeded, PolynomialError,
    MultivariatePolynomialError,
    ExactQuotientFailed,
    PolificationFailed,
    ComputationFailed,
    GeneratorsError,
)
from sympy.polys.polyutils import (
    basic_from_dict,
    _sort_gens,
    _unify_gens,
    _dict_reorder,
    _dict_from_expr,
    _parallel_dict_from_expr,
)

def _poly_from_expr(expr, opt):
    """Construct a polynomial from an expression. """
    orig, expr = expr, sympify(expr)

    if not isinstance(expr, Basic):
        raise PolificationFailed(opt, orig, expr)
    elif expr.is_Poly:
        poly = expr.__class__._from_poly(expr, opt)

        opt.gens = poly.gens
        opt.domain = poly.domain

        if opt.polys is None:
            opt.polys = True

        return poly, opt
    elif opt.expand:
        expr = expr.expand()

    rep, opt = _dict_from_expr(expr, opt)
    if not opt.gens:
        raise PolificationFailed(opt, orig, expr)

    monoms, coeffs = list(zip(*list(rep.items())))
    domain = opt.domain

    if domain is None:
        opt.domain, coeffs = construct_domain(coeffs, opt=opt)
    else:
        coeffs = list(map(domain.from_sympy, coeffs))

    rep = dict(list(zip(monoms, coeffs)))
    poly = Poly._from_dict(rep, opt)

    if opt.polys is None:
        opt.polys = False

    return poly, opt
