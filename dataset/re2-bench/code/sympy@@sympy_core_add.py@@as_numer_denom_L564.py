from typing import TYPE_CHECKING, ClassVar
from collections import defaultdict
from functools import reduce
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .intfunc import ilcm, igcd
from .expr import Expr
from .mul import Mul, _keep_coeff, _unevaluated_Mul
from .numbers import Rational

class Add(Expr, AssocOp):
    """
    Expression representing addition operation for algebraic group.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Add()`` must be ``Expr``. Infix operator ``+``
    on most scalar objects in SymPy calls this class.

    Another use of ``Add()`` is to represent the structure of abstract
    addition so that its arguments can be substituted to return different
    class. Refer to examples section for this.

    ``Add()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Add(x, Add(y, z))`` -> ``Add(x, y, z)``

    2. Identity removing
        ``Add(x, 0, y)`` -> ``Add(x, y)``

    3. Coefficient collecting by ``.as_coeff_Mul()``
        ``Add(x, 2*x)`` -> ``Mul(3, x)``

    4. Term sorting
        ``Add(y, x, 2)`` -> ``Add(2, x, y)``

    If no argument is passed, identity element 0 is returned. If single
    element is passed, that element is returned.

    Note that ``Add(*args)`` is more efficient than ``sum(args)`` because
    it flattens the arguments. ``sum(a, b, c, ...)`` recursively adds the
    arguments as ``a + (b + (c + ...))``, which has quadratic complexity.
    On the other hand, ``Add(a, b, c, d)`` does not assume nested
    structure, making the complexity linear.

    Since addition is group operation, every argument should have the
    same :obj:`sympy.core.kind.Kind()`.

    Examples
    ========

    >>> from sympy import Add, I
    >>> from sympy.abc import x, y
    >>> Add(x, 1)
    x + 1
    >>> Add(x, x)
    2*x
    >>> 2*x**2 + 3*x + I*y + 2*y + 2*x/5 + 1.0*y + 1
    2*x**2 + 17*x/5 + 3.0*y + I*y + 1

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Add(1, 2, evaluate=False)
    1 + 2
    >>> Add(x, x, evaluate=False)
    x + x

    ``Add()`` also represents the general structure of addition operation.

    >>> from sympy import MatrixSymbol
    >>> A,B = MatrixSymbol('A', 2,2), MatrixSymbol('B', 2,2)
    >>> expr = Add(x,y).subs({x:A, y:B})
    >>> expr
    A + B
    >>> type(expr)
    <class 'sympy.matrices.expressions.matadd.MatAdd'>

    Note that the printers do not display in args order.

    >>> Add(x, 1)
    x + 1
    >>> Add(x, 1).args
    (1, x)

    See Also
    ========

    MatAdd

    """
    __slots__ = ()
    is_Add = True
    _args_type = Expr
    identity: ClassVar[Expr]
    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr:
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        """
        Decomposes an expression to its numerator part and its
        denominator part.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> (x*y/z).as_numer_denom()
        (x*y, z)
        >>> (x*(y + 1)/y**7).as_numer_denom()
        (x*(y + 1), y**7)

        See Also
        ========

        sympy.core.expr.Expr.as_numer_denom
        """
        content, expr = self.primitive()
        if not isinstance(expr, Add):
            return Mul(content, expr, evaluate=False).as_numer_denom()
        ncon, dcon = content.as_numer_denom()
        nd = defaultdict(list)
        for f in expr.args:
            ni, di = f.as_numer_denom()
            nd[di].append(ni)
        if len(nd) == 1:
            d, n = nd.popitem()
            return (self.func(*[_keep_coeff(ncon, ni) for ni in n]), _keep_coeff(dcon, d))
        nd2 = {d: self.func(*n) if len(n) > 1 else n[0] for d, n in nd.items()}
        denoms, numers = [list(i) for i in zip(*iter(nd2.items()))]
        n, d = (self.func(*[Mul(*denoms[:i] + [numers[i]] + denoms[i + 1:]) for i in range(len(numers))]), Mul(*denoms))
        return (_keep_coeff(ncon, n), _keep_coeff(dcon, d))
    _eval_is_real = lambda self: _fuzzy_group((a.is_real for a in self.args), quick_exit=True)
    _eval_is_extended_real = lambda self: _fuzzy_group((a.is_extended_real for a in self.args), quick_exit=True)
    _eval_is_complex = lambda self: _fuzzy_group((a.is_complex for a in self.args), quick_exit=True)
    _eval_is_antihermitian = lambda self: _fuzzy_group((a.is_antihermitian for a in self.args), quick_exit=True)
    _eval_is_finite = lambda self: _fuzzy_group((a.is_finite for a in self.args), quick_exit=True)
    _eval_is_hermitian = lambda self: _fuzzy_group((a.is_hermitian for a in self.args), quick_exit=True)
    _eval_is_integer = lambda self: _fuzzy_group((a.is_integer for a in self.args), quick_exit=True)
    _eval_is_rational = lambda self: _fuzzy_group((a.is_rational for a in self.args), quick_exit=True)
    _eval_is_algebraic = lambda self: _fuzzy_group((a.is_algebraic for a in self.args), quick_exit=True)
    _eval_is_commutative = lambda self: _fuzzy_group((a.is_commutative for a in self.args))

    def primitive(self):
        """
        Return ``(R, self/R)`` where ``R``` is the Rational GCD of ``self```.

        ``R`` is collected only from the leading coefficient of each term.

        Examples
        ========

        >>> from sympy.abc import x, y

        >>> (2*x + 4*y).primitive()
        (2, x + 2*y)

        >>> (2*x/3 + 4*y/9).primitive()
        (2/9, 3*x + 2*y)

        >>> (2*x/3 + 4.2*y).primitive()
        (1/3, 2*x + 12.6*y)

        No subprocessing of term factors is performed:

        >>> ((2 + 2*x)*x + 2).primitive()
        (1, x*(2*x + 2) + 2)

        Recursive processing can be done with the ``as_content_primitive()``
        method:

        >>> ((2 + 2*x)*x + 2).as_content_primitive()
        (2, x*(x + 1) + 1)

        See also: primitive() function in polytools.py

        """
        terms = []
        inf = False
        for a in self.args:
            c, m = a.as_coeff_Mul()
            if not c.is_Rational:
                c = S.One
                m = a
            inf = inf or m is S.ComplexInfinity
            terms.append((c.p, c.q, m))
        if not inf:
            ngcd = reduce(igcd, [t[0] for t in terms], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms], 1)
        else:
            ngcd = reduce(igcd, [t[0] for t in terms if t[1]], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms if t[1]], 1)
        if ngcd == dlcm == 1:
            return (S.One, self)
        if not inf:
            for i, (p, q, term) in enumerate(terms):
                terms[i] = _keep_coeff(Rational(p // ngcd * (dlcm // q)), term)
        else:
            for i, (p, q, term) in enumerate(terms):
                if q:
                    terms[i] = _keep_coeff(Rational(p // ngcd * (dlcm // q)), term)
                else:
                    terms[i] = _keep_coeff(Rational(p, q), term)
        if terms[0].is_Number or terms[0] is S.ComplexInfinity:
            c = terms.pop(0)
        else:
            c = None
        _addsort(terms)
        if c:
            terms.insert(0, c)
        return (Rational(ngcd, dlcm), self._new_rawargs(*terms))
