from typing import TYPE_CHECKING, ClassVar, overload, Literal
from .sympify import sympify
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .intfunc import integer_nthroot, trailing
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .kind import KindDispatcher
from .power import Pow
from sympy.functions.elementary.complexes import Abs, im, re
from .function import expand_mul
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.complexes import sign

class Mul(Expr, AssocOp):
    """
    Expression representing multiplication operation for algebraic field.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Mul()`` must be ``Expr``. Infix operator ``*``
    on most scalar objects in SymPy calls this class.

    Another use of ``Mul()`` is to represent the structure of abstract
    multiplication so that its arguments can be substituted to return
    different class. Refer to examples section for this.

    ``Mul()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Mul(x, Mul(y, z))`` -> ``Mul(x, y, z)``

    2. Identity removing
        ``Mul(x, 1, y)`` -> ``Mul(x, y)``

    3. Exponent collecting by ``.as_base_exp()``
        ``Mul(x, x**2)`` -> ``Pow(x, 3)``

    4. Term sorting
        ``Mul(y, x, 2)`` -> ``Mul(2, x, y)``

    Since multiplication can be vector space operation, arguments may
    have the different :obj:`sympy.core.kind.Kind()`. Kind of the
    resulting object is automatically inferred.

    Examples
    ========

    >>> from sympy import Mul
    >>> from sympy.abc import x, y
    >>> Mul(x, 1)
    x
    >>> Mul(x, x)
    x**2

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Mul(1, 2, evaluate=False)
    1*2
    >>> Mul(x, x, evaluate=False)
    x*x

    ``Mul()`` also represents the general structure of multiplication
    operation.

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2,2)
    >>> expr = Mul(x,y).subs({y:A})
    >>> expr
    x*A
    >>> type(expr)
    <class 'sympy.matrices.expressions.matmul.MatMul'>

    See Also
    ========

    MatMul

    """
    __slots__ = ()
    is_Mul = True
    _args_type = Expr
    _kind_dispatcher = KindDispatcher('Mul_kind_dispatcher', commutative=True)
    identity: ClassVar[Expr]
    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr:
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    def _eval_power(self, expt):
        cargs, nc = self.args_cnc(split_1=False)
        if expt.is_Integer:
            return Mul(*[Pow(b, expt, evaluate=False) for b in cargs]) * Pow(Mul._from_args(nc), expt, evaluate=False)
        if expt.is_Rational and expt.q == 2:
            if self.is_imaginary:
                a = self.as_real_imag()[1]
                if a.is_Rational:
                    n, d = abs(a / 2).as_numer_denom()
                    n, t = integer_nthroot(n, 2)
                    if t:
                        d, t = integer_nthroot(d, 2)
                        if t:
                            from sympy.functions.elementary.complexes import sign
                            r = sympify(n) / d
                            return _unevaluated_Mul(r ** expt.p, (1 + sign(a) * S.ImaginaryUnit) ** expt.p)
        p = Pow(self, expt, evaluate=False)
        if expt.is_Rational or expt.is_Float:
            return p._eval_expand_power_base()
        return p

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.complexes import Abs, im, re
        other = []
        coeffr = []
        coeffi = []
        addterms = S.One
        for a in self.args:
            r, i = a.as_real_imag()
            if i.is_zero:
                coeffr.append(r)
            elif r.is_zero:
                coeffi.append(i * S.ImaginaryUnit)
            elif a.is_commutative:
                aconj = a.conjugate() if other else None
                for i, x in enumerate(other):
                    if x == aconj:
                        coeffr.append(Abs(x) ** 2)
                        del other[i]
                        break
                else:
                    if a.is_Add:
                        addterms *= a
                    else:
                        other.append(a)
            else:
                other.append(a)
        m = self.func(*other)
        if hints.get('ignore') == m:
            return
        if len(coeffi) % 2:
            imco = im(coeffi.pop(0))
        else:
            imco = S.Zero
        reco = self.func(*coeffr + coeffi)
        r, i = (reco * re(m), reco * im(m))
        if addterms == 1:
            if m == 1:
                if imco.is_zero:
                    return (reco, S.Zero)
                else:
                    return (S.Zero, reco * imco)
            if imco is S.Zero:
                return (r, i)
            return (-imco * i, imco * r)
        from .function import expand_mul
        addre, addim = expand_mul(addterms, deep=False).as_real_imag()
        if imco is S.Zero:
            return (r * addre - i * addim, i * addre + r * addim)
        else:
            r, i = (-imco * i, imco * r)
            return (r * addre - i * addim, r * addim + i * addre)
    _eval_is_commutative = lambda self: _fuzzy_group((a.is_commutative for a in self.args))

    def args_cnc(self, cset=False, warn=True, split_1=True):
        """Return [commutative factors, non-commutative factors] of self.

        Explanation
        ===========

        self is treated as a Mul and the ordering of the factors is maintained.
        If ``cset`` is True the commutative factors will be returned in a set.
        If there were repeated factors (as may happen with an unevaluated Mul)
        then an error will be raised unless it is explicitly suppressed by
        setting ``warn`` to False.

        Note: -1 is always separated from a Number unless split_1 is False.

        Examples
        ========

        >>> from sympy import symbols, oo
        >>> A, B = symbols('A B', commutative=0)
        >>> x, y = symbols('x y')
        >>> (-2*x*y).args_cnc()
        [[-1, 2, x, y], []]
        >>> (-2.5*x).args_cnc()
        [[-1, 2.5, x], []]
        >>> (-2*x*A*B*y).args_cnc()
        [[-1, 2, x, y], [A, B]]
        >>> (-2*x*A*B*y).args_cnc(split_1=False)
        [[-2, x, y], [A, B]]
        >>> (-2*x*y).args_cnc(cset=True)
        [{-1, 2, x, y}, []]

        The arg is always treated as a Mul:

        >>> (-2 + x + A).args_cnc()
        [[], [x - 2 + A]]
        >>> (-oo).args_cnc() # -oo is a singleton
        [[-1, oo], []]
        """
        args = list(Mul.make_args(self))
        for i, mi in enumerate(args):
            if not mi.is_commutative:
                c = args[:i]
                nc = args[i:]
                break
        else:
            c = args
            nc = []
        if c and split_1 and (c[0].is_Number and c[0].is_extended_negative and (c[0] is not S.NegativeOne)):
            c[:1] = [S.NegativeOne, -c[0]]
        if cset:
            clen = len(c)
            c = set(c)
            if clen and warn and (len(c) != clen):
                raise ValueError('repeated commutative arguments: %s' % [ci for ci in c if list(self.args).count(ci) > 1])
        return [c, nc]
