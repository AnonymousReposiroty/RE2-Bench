from typing import TYPE_CHECKING, ClassVar, overload, Literal
from collections import defaultdict
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .kind import KindDispatcher
from .power import Pow
from sympy.simplify.radsimp import fraction
from sympy.simplify.radsimp import fraction
from sympy.simplify.radsimp import fraction
from sympy.functions.elementary.complexes import sign
from sympy.ntheory.factor_ import multiplicity
from sympy.simplify.powsimp import powdenest
from sympy.simplify.radsimp import fraction
from sympy.functions.elementary.exponential import exp
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
    _eval_is_commutative = lambda self: _fuzzy_group((a.is_commutative for a in self.args))

    def _eval_subs(self, old, new):
        from sympy.functions.elementary.complexes import sign
        from sympy.ntheory.factor_ import multiplicity
        from sympy.simplify.powsimp import powdenest
        from sympy.simplify.radsimp import fraction
        if not old.is_Mul:
            return None
        if old.args[0].is_Number and old.args[0] < 0:
            if self.args[0].is_Number:
                if self.args[0] < 0:
                    return self._subs(-old, -new)
                return None

        def base_exp(a):
            from sympy.functions.elementary.exponential import exp
            if a.is_Pow or isinstance(a, exp):
                return a.as_base_exp()
            return (a, S.One)

        def breakup(eq):
            """break up powers of eq when treated as a Mul:
                   b**(Rational*e) -> b**e, Rational
                commutatives come back as a dictionary {b**e: Rational}
                noncommutatives come back as a list [(b**e, Rational)]
            """
            c, nc = (defaultdict(int), [])
            for a in Mul.make_args(eq):
                a = powdenest(a)
                b, e = base_exp(a)
                if e is not S.One:
                    co, _ = e.as_coeff_mul()
                    b = Pow(b, e / co)
                    e = co
                if a.is_commutative:
                    c[b] += e
                else:
                    nc.append([b, e])
            return (c, nc)

        def rejoin(b, co):
            """
            Put rational back with exponent; in general this is not ok, but
            since we took it from the exponent for analysis, it's ok to put
            it back.
            """
            b, e = base_exp(b)
            return Pow(b, e * co)

        def ndiv(a, b):
            """if b divides a in an extractive way (like 1/4 divides 1/2
            but not vice versa, and 2/5 does not divide 1/3) then return
            the integer number of times it divides, else return 0.
            """
            if not b.q % a.q or not a.q % b.q:
                return int(a / b)
            return 0
        rv = None
        n, d = fraction(self)
        self2 = self
        if d is not S.One:
            self2 = n._subs(old, new) / d._subs(old, new)
            if not self2.is_Mul:
                return self2._subs(old, new)
            if self2 != self:
                rv = self2
        co_self = self2.args[0]
        co_old = old.args[0]
        co_xmul = None
        if co_old.is_Rational and co_self.is_Rational:
            if co_old != co_self:
                co_xmul = co_self.extract_multiplicatively(co_old)
        elif co_old.is_Rational:
            return rv
        c, nc = breakup(self2)
        old_c, old_nc = breakup(old)
        if co_xmul and co_xmul.is_Rational and (abs(co_old) != 1):
            mult = S(multiplicity(abs(co_old), co_self))
            c.pop(co_self)
            if co_old in c:
                c[co_old] += mult
            else:
                c[co_old] = mult
            co_residual = co_self / co_old ** mult
        else:
            co_residual = 1
        ok = True
        if len(old_nc) > len(nc):
            ok = False
        elif len(old_c) > len(c):
            ok = False
        elif {i[0] for i in old_nc}.difference({i[0] for i in nc}):
            ok = False
        elif set(old_c).difference(set(c)):
            ok = False
        elif any((sign(c[b]) != sign(old_c[b]) for b in old_c)):
            ok = False
        if not ok:
            return rv
        if not old_c:
            cdid = None
        else:
            rat = []
            for b, old_e in old_c.items():
                c_e = c[b]
                rat.append(ndiv(c_e, old_e))
                if not rat[-1]:
                    return rv
            cdid = min(rat)
        if not old_nc:
            ncdid = None
            for i in range(len(nc)):
                nc[i] = rejoin(*nc[i])
        else:
            ncdid = 0
            take = len(old_nc)
            limit = cdid or S.Infinity
            failed = []
            i = 0
            while limit and i + take <= len(nc):
                hit = False
                rat = []
                for j in range(take):
                    if nc[i + j][0] != old_nc[j][0]:
                        break
                    elif j == 0 or j == take - 1:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif nc[i + j][1] != old_nc[j][1]:
                        break
                    else:
                        rat.append(1)
                else:
                    ndo = min(rat)
                    if ndo:
                        if take == 1:
                            if cdid:
                                ndo = min(cdid, ndo)
                            nc[i] = Pow(new, ndo) * rejoin(nc[i][0], nc[i][1] - ndo * old_nc[0][1])
                        else:
                            ndo = 1
                            l = rejoin(nc[i][0], nc[i][1] - ndo * old_nc[0][1])
                            mid = new
                            ir = i + take - 1
                            r = (nc[ir][0], nc[ir][1] - ndo * old_nc[-1][1])
                            if r[1]:
                                if i + take < len(nc):
                                    nc[i:i + take] = [l * mid, r]
                                else:
                                    r = rejoin(*r)
                                    nc[i:i + take] = [l * mid * r]
                            else:
                                nc[i:i + take] = [l * mid]
                        limit -= ndo
                        ncdid += ndo
                        hit = True
                if not hit:
                    failed.append(i)
                i += 1
            else:
                if not ncdid:
                    return rv
                failed.extend(range(i, len(nc)))
                for i in failed:
                    nc[i] = rejoin(*nc[i]).subs(old, new)
        if cdid is None:
            do = ncdid
        elif ncdid is None:
            do = cdid
        else:
            do = min(ncdid, cdid)
        margs = []
        for b in c:
            if b in old_c:
                e = c[b] - old_c[b] * do
                margs.append(rejoin(b, e))
            else:
                margs.append(rejoin(b.subs(old, new), c[b]))
        if cdid and (not ncdid):
            margs = [Pow(new, cdid)] + margs
        return co_residual * self2.func(*margs) * self2.func(*nc)
