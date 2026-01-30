from typing import TYPE_CHECKING, Optional, overload, Literal, Any, cast, Callable
from sympy.core import (
    S, Expr, Add, Tuple
)
from sympy.core.basic import Basic
from sympy.core.sympify import sympify, _sympify
from sympy.polys.domains.domain import Domain
from sympy.polys.polyclasses import DMP, DMF, ANP
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
from sympy.utilities import group, public, filldedent

@public
class Poly(Basic):
    """
    Generic class for representing and operating on polynomial expressions.

    See :ref:`polys-docs` for general documentation.

    Poly is a subclass of Basic rather than Expr but instances can be
    converted to Expr with the :py:meth:`~.Poly.as_expr` method.

    .. deprecated:: 1.6

       Combining Poly with non-Poly objects in binary operations is
       deprecated. Explicitly convert both objects to either Poly or Expr
       first. See :ref:`deprecated-poly-nonpoly-binary-operations`.

    Examples
    ========

    >>> from sympy import Poly
    >>> from sympy.abc import x, y

    Create a univariate polynomial:

    >>> Poly(x*(x**2 + x - 1)**2)
    Poly(x**5 + 2*x**4 - x**3 - 2*x**2 + x, x, domain='ZZ')

    Create a univariate polynomial with specific domain:

    >>> from sympy import sqrt
    >>> Poly(x**2 + 2*x + sqrt(3), domain='R')
    Poly(1.0*x**2 + 2.0*x + 1.73205080756888, x, domain='RR')

    Create a multivariate polynomial:

    >>> Poly(y*x**2 + x*y + 1)
    Poly(x**2*y + x*y + 1, x, y, domain='ZZ')

    Create a univariate polynomial, where y is a constant:

    >>> Poly(y*x**2 + x*y + 1,x)
    Poly(y*x**2 + y*x + 1, x, domain='ZZ[y]')

    You can evaluate the above polynomial as a function of y:

    >>> Poly(y*x**2 + x*y + 1,x).eval(2)
    6*y + 1

    See Also
    ========

    sympy.core.expr.Expr

    """
    __slots__ = ('rep', 'gens')
    is_commutative = True
    is_Poly = True
    _op_priority = 10.001
    rep: DMP
    gens: tuple[Expr, ...]

    @classmethod
    def new(cls, rep, *gens):
        """Construct :class:`Poly` instance from raw representation. """
        if not isinstance(rep, DMP):
            raise PolynomialError('invalid polynomial representation: %s' % rep)
        elif rep.lev != len(gens) - 1:
            raise PolynomialError('invalid arguments: %s, %s' % (rep, gens))
        obj = Basic.__new__(cls)
        obj.rep = rep
        obj.gens = gens
        return obj

    def _unify(f, g: Poly | Expr | complex) -> tuple[Domain, Callable[[DMP], Poly], DMP, DMP]:
        gs = cast('Poly | Expr', sympify(g))
        if not isinstance(gs, Poly):
            try:
                g_coeff = f.rep.dom.from_sympy(gs)
            except CoercionFailed:
                raise UnificationFailed('Cannot unify %s with %s' % (f, gs))
            else:
                return (f.rep.dom, f.per, f.rep, f.rep.ground_new(g_coeff))
        if isinstance(f.rep, DMP) and isinstance(gs.rep, DMP):
            gens = _unify_gens(f.gens, gs.gens)
            dom, lev = (f.rep.dom.unify(gs.rep.dom, gens), len(gens) - 1)
            if f.gens != gens:
                f_monoms, f_coeffs = _dict_reorder(f.rep.to_dict(), f.gens, gens)
                if f.rep.dom != dom:
                    f_coeffs = [dom.convert(c, f.rep.dom) for c in f_coeffs]
                F = DMP.from_dict(dict(list(zip(f_monoms, f_coeffs))), lev, dom)
            else:
                F = f.rep.convert(dom)
            if gs.gens != gens:
                g_monoms, g_coeffs = _dict_reorder(gs.rep.to_dict(), gs.gens, gens)
                if gs.rep.dom != dom:
                    g_coeffs = [dom.convert(c, gs.rep.dom) for c in g_coeffs]
                G = DMP.from_dict(dict(list(zip(g_monoms, g_coeffs))), lev, dom)
            else:
                G = gs.rep.convert(dom)
        else:
            raise UnificationFailed('Cannot unify %s with %s' % (f, gs))
        cls = f.__class__

        def per(rep, dom=dom, gens=gens, remove=None):
            if remove is not None:
                gens = gens[:remove] + gens[remove + 1:]
                if not gens:
                    return dom.to_sympy(rep)
            return cls.new(rep, *gens)
        return (dom, per, F, G)
    _eval_derivative = diff
