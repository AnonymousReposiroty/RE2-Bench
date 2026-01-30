from typing import TYPE_CHECKING, Optional, overload, Literal, Any, cast, Callable
from sympy.core import (
    S, Expr, Add, Tuple
)
from sympy.core.basic import Basic
from sympy.core.sympify import sympify, _sympify
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains.domainelement import DomainElement
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
from sympy.utilities.iterables import iterable, sift
from typing import Self

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

    def __new__(cls, rep, *gens, **args) -> Self:
        """Create a new polynomial instance out of something useful. """
        opt = options.build_options(gens, args)
        if 'order' in opt:
            raise NotImplementedError("'order' keyword is not implemented yet")
        if isinstance(rep, (DMP, DMF, ANP, DomainElement)):
            return cls._from_domain_element(rep, opt)
        elif iterable(rep, exclude=str):
            if isinstance(rep, dict):
                return cls._from_dict(rep, opt)
            else:
                return cls._from_list(list(rep), opt)
        else:
            rep = sympify(rep, evaluate=type(rep) is not str)
            if rep.is_Poly:
                return cls._from_poly(rep, opt)
            else:
                return cls._from_expr(rep, opt)

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

    @classmethod
    def _from_dict(cls, rep: dict[tuple[int, ...], Any] | dict[int, Any], opt):
        """Construct a polynomial from a ``dict``. """
        gens = opt.gens
        if not gens:
            raise GeneratorsNeeded("Cannot initialize from 'dict' without generators")
        level = len(gens) - 1
        domain = opt.domain
        if domain is None:
            domain, rep_d = construct_domain(rep, opt=opt)
        else:
            convert = domain.convert
            rep_d = {monom: convert(coeff) for monom, coeff in rep.items()}
        n = None
        for n in rep_d:
            break
        if isinstance(n, int):
            raw_dict = cast(dict[int, Any], rep_d)
            return cls.new(DMP.from_raw_dict(raw_dict, domain), *gens)
        else:
            multi_dict = cast(dict[tuple[int, ...], Any], rep_d)
            return cls.new(DMP.from_dict(multi_dict, level, domain), *gens)

    @classmethod
    def _from_list(cls, rep, opt):
        """Construct a polynomial from a ``list``. """
        gens = opt.gens
        if not gens:
            raise GeneratorsNeeded("Cannot initialize from 'list' without generators")
        elif len(gens) != 1:
            raise MultivariatePolynomialError("'list' representation not supported")
        level = len(gens) - 1
        domain = opt.domain
        if domain is None:
            domain, rep = construct_domain(rep, opt=opt)
        else:
            rep = list(map(domain.convert, rep))
        return cls.new(DMP.from_list(rep, level, domain), *gens)

    @classmethod
    def _from_poly(cls, rep, opt):
        """Construct a polynomial from a polynomial. """
        if cls != rep.__class__:
            rep = cls.new(rep.rep, *rep.gens)
        gens = opt.gens
        field = opt.field
        domain = opt.domain
        if gens and rep.gens != gens:
            if set(rep.gens) != set(gens):
                return cls._from_expr(rep.as_expr(), opt)
            else:
                rep = rep.reorder(*gens)
        if 'domain' in opt and domain:
            rep = rep.set_domain(domain)
        elif field is True:
            rep = rep.to_field()
        return rep

    @classmethod
    def _from_expr(cls, rep, opt):
        """Construct a polynomial from an expression. """
        rep, opt = _dict_from_expr(rep, opt)
        return cls._from_dict(rep, opt)

    @classmethod
    def _from_domain_element(cls, rep, opt):
        gens = opt.gens
        domain = opt.domain
        level = len(gens) - 1
        rep = [domain.convert(rep)]
        return cls.new(DMP.from_list(rep, level, domain), *gens)
    _eval_derivative = diff
