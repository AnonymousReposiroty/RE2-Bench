from typing import TYPE_CHECKING, overload, Literal
from collections.abc import Iterable, Mapping
from functools import reduce
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from sympy.utilities.misc import as_int, func_name, filldedent
from sympy.utilities.iterables import has_variety, _sift_true_false
from .mul import Mul
from .add import Add
from typing import Any, Hashable
from typing_extensions import Self
from sympy.functions.elementary.complexes import conjugate as c
from .symbol import Symbol
from .add import _unevaluated_Add
from .mul import _unevaluated_Mul
from .mul import _unevaluated_Mul
from .add import _unevaluated_Add
from .symbol import Dummy, Symbol
from .symbol import Dummy, Symbol

@sympify_method_args
class Expr(Basic, EvalfMixin):
    """
    Base class for algebraic expressions.

    Explanation
    ===========

    Everything that requires arithmetic operations to be defined
    should subclass this class, instead of Basic (which should be
    used only for argument storage and expression manipulation, i.e.
    pattern matching, substitutions, etc).

    If you want to override the comparisons of expressions:
    Should use _eval_is_ge for inequality, or _eval_is_eq, with multiple dispatch.
    _eval_is_ge return true if x >= y, false if x < y, and None if the two types
    are not comparable or the comparison is indeterminate

    See Also
    ========

    sympy.core.basic.Basic
    """
    __slots__: tuple[str, ...] = ()
    if TYPE_CHECKING:

        def __new__(cls, *args: Basic) -> Self:
            ...

        @overload
        def subs(self, arg1: Mapping[Basic | complex, Expr | complex], arg2: None=None) -> Expr:
            ...

        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Expr | complex]], arg2: None=None, **kwargs: Any) -> Expr:
            ...

        @overload
        def subs(self, arg1: Expr | complex, arg2: Expr | complex) -> Expr:
            ...

        @overload
        def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic:
            ...

        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic:
            ...

        @overload
        def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic:
            ...

        def subs(self, arg1: Mapping[Basic | complex, Basic | complex] | Basic | complex, arg2: Basic | complex | None=None, **kwargs: Any) -> Basic:
            ...

        def simplify(self, **kwargs) -> Expr:
            ...

        def evalf(self, n: int | None=15, subs: dict[Basic, Basic | float] | None=None, maxn: int=100, chop: bool | int=False, strict: bool=False, quad: str | None=None, verbose: bool=False) -> Expr:
            ...
        n = evalf
    is_scalar = True
    _op_priority = 10.0

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

    def coeff(self, x: Expr | complex, n=1, right=False, _first=True):
        """
        Returns the coefficient from the term(s) containing ``x**n``. If ``n``
        is zero then all terms independent of ``x`` will be returned.

        Explanation
        ===========

        When ``x`` is noncommutative, the coefficient to the left (default) or
        right of ``x`` can be returned. The keyword 'right' is ignored when
        ``x`` is commutative.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.abc import x, y, z

        You can select terms that have an explicit negative in front of them:

        >>> (-x + 2*y).coeff(-1)
        x
        >>> (x - 2*y).coeff(-1)
        2*y

        You can select terms with no Rational coefficient:

        >>> (x + 2*y).coeff(1)
        x
        >>> (3 + 2*x + 4*x**2).coeff(1)
        0

        You can select terms independent of x by making n=0; in this case
        expr.as_independent(x)[0] is returned (and 0 will be returned instead
        of None):

        >>> (3 + 2*x + 4*x**2).coeff(x, 0)
        3
        >>> eq = ((x + 1)**3).expand() + 1
        >>> eq
        x**3 + 3*x**2 + 3*x + 2
        >>> [eq.coeff(x, i) for i in reversed(range(4))]
        [1, 3, 3, 2]
        >>> eq -= 2
        >>> [eq.coeff(x, i) for i in reversed(range(4))]
        [1, 3, 3, 0]

        You can select terms that have a numerical term in front of them:

        >>> (-x - 2*y).coeff(2)
        -y
        >>> from sympy import sqrt
        >>> (x + sqrt(2)*x).coeff(sqrt(2))
        x

        The matching is exact:

        >>> (3 + 2*x + 4*x**2).coeff(x)
        2
        >>> (3 + 2*x + 4*x**2).coeff(x**2)
        4
        >>> (3 + 2*x + 4*x**2).coeff(x**3)
        0
        >>> (z*(x + y)**2).coeff((x + y)**2)
        z
        >>> (z*(x + y)**2).coeff(x + y)
        0

        In addition, no factoring is done, so 1 + z*(1 + y) is not obtained
        from the following:

        >>> (x + z*(x + x*y)).coeff(x)
        1

        If such factoring is desired, factor_terms can be used first:

        >>> from sympy import factor_terms
        >>> factor_terms(x + z*(x + x*y)).coeff(x)
        z*(y + 1) + 1

        >>> n, m, o = symbols('n m o', commutative=False)
        >>> n.coeff(n)
        1
        >>> (3*n).coeff(n)
        3
        >>> (n*m + m*n*m).coeff(n) # = (1 + m)*n*m
        1 + m
        >>> (n*m + m*n*m).coeff(n, right=True) # = (1 + m)*n*m
        m

        If there is more than one possible coefficient 0 is returned:

        >>> (n*m + m*n).coeff(n)
        0

        If there is only one possible coefficient, it is returned:

        >>> (n*m + x*m*n).coeff(m*n)
        x
        >>> (n*m + x*m*n).coeff(m*n, right=1)
        1

        See Also
        ========

        as_coefficient: separate the expression into a coefficient and factor
        as_coeff_Add: separate the additive constant from an expression
        as_coeff_Mul: separate the multiplicative constant from an expression
        as_independent: separate x-dependent terms/factors from others
        sympy.polys.polytools.Poly.coeff_monomial: efficiently find the single coefficient of a monomial in Poly
        sympy.polys.polytools.Poly.nth: like coeff_monomial but powers of monomial terms are used
        """
        xe = sympify(x)
        if not isinstance(xe, Basic):
            return S.Zero
        n = as_int(n)
        if not xe:
            return S.Zero
        if xe == self:
            if n == 1:
                return S.One
            return S.Zero
        co2: list[Expr]
        if xe is S.One:
            co2 = [a for a in Add.make_args(self) if a.as_coeff_Mul()[0] is S.One]
            if not co2:
                return S.Zero
            return Add(*co2)
        if n == 0:
            if xe.is_Add and self.is_Add:
                c = self.coeff(xe, right=right)
                if not c:
                    return S.Zero
                if not right:
                    return self - Add(*[a * xe for a in Add.make_args(c)])
                return self - Add(*[xe * a for a in Add.make_args(c)])
            return self.as_independent(xe, as_Add=True)[0]
        xe = xe ** n

        def incommon(l1, l2):
            if not l1 or not l2:
                return []
            n = min(len(l1), len(l2))
            for i in range(n):
                if l1[i] != l2[i]:
                    return l1[:i]
            return l1[:]

        def find(l, sub, first=True):
            """ Find where list sub appears in list l. When ``first`` is True
            the first occurrence from the left is returned, else the last
            occurrence is returned. Return None if sub is not in l.

            Examples
            ========

            >> l = range(5)*2
            >> find(l, [2, 3])
            2
            >> find(l, [2, 3], first=0)
            7
            >> find(l, [2, 4])
            None

            """
            if not sub or not l or len(sub) > len(l):
                return None
            n = len(sub)
            if not first:
                l.reverse()
                sub.reverse()
            for i in range(len(l) - n + 1):
                if all((l[i + j] == sub[j] for j in range(n))):
                    break
            else:
                i = None
            if not first:
                l.reverse()
                sub.reverse()
            if i is not None and (not first):
                i = len(l) - (i + n)
            return i
        co2 = []
        co: list[tuple[set[Expr], list[Expr]]] = []
        args = Add.make_args(self)
        self_c = self.is_commutative
        x_c = xe.is_commutative
        if self_c and (not x_c):
            return S.Zero
        if _first and self.is_Add and (not self_c) and (not x_c):
            xargs = Mul.make_args(xe)
            d = Add(*[i for i in Add.make_args(self.as_independent(xe)[1]) if all((xi in Mul.make_args(i) for xi in xargs))])
            rv = d.coeff(xe, right=right, _first=False)
            if not rv.is_Add or not right:
                return rv
            c_part, nc_part = zip(*[i.args_cnc() for i in rv.args])
            if has_variety(c_part):
                return rv
            return Add(*[Mul._from_args(i) for i in nc_part])
        one_c = self_c or x_c
        xargs, nx = xe.args_cnc(cset=True, warn=bool(not x_c))
        for a in args:
            margs, nc = a.args_cnc(cset=True, warn=bool(not self_c))
            if nc is None:
                nc = []
            if len(xargs) > len(margs):
                continue
            resid = margs.difference(xargs)
            if len(resid) + len(xargs) == len(margs):
                if one_c:
                    co2.append(Mul(*list(resid) + nc))
                else:
                    co.append((resid, nc))
        if one_c:
            if co2 == []:
                return S.Zero
            elif co2:
                return Add(*co2)
        else:
            if not co:
                return S.Zero
            if all((n == co[0][1] for r, n in co)):
                ii = find(co[0][1], nx, right)
                if ii is not None:
                    if not right:
                        return Mul(Add(*[Mul(*r) for r, c in co]), Mul(*co[0][1][:ii]))
                    else:
                        return Mul(*co[0][1][ii + len(nx):])
            beg = reduce(incommon, (n[1] for n in co))
            if beg:
                ii = find(beg, nx, right)
                if ii is not None:
                    if not right:
                        gcdc = co[0][0]
                        for i in range(1, len(co)):
                            gcdc = gcdc.intersection(co[i][0])
                            if not gcdc:
                                break
                        return Mul(*list(gcdc) + beg[:ii])
                    else:
                        m = ii + len(nx)
                        return Add(*[Mul(*list(r) + n[m:]) for r, n in co])
            end = list(reversed(reduce(incommon, (list(reversed(n[1])) for n in co))))
            if end:
                ii = find(end, nx, right)
                if ii is not None:
                    if not right:
                        return Add(*[Mul(*list(r) + n[:-len(end) + ii]) for r, n in co])
                    else:
                        return Mul(*end[ii + len(nx):])
            hit = None
            for i, (r, n) in enumerate(co):
                ii = find(n, nx, right)
                if ii is not None:
                    if not hit:
                        hit = (ii, r, n)
                    else:
                        break
            else:
                if hit:
                    ii, r, n = hit
                    if not right:
                        return Mul(*list(r) + n[:ii])
                    else:
                        return Mul(*n[ii + len(nx):])
            return S.Zero

    def as_independent(self, *deps: Basic | type[Basic], as_Add: bool | None=None, strict: bool=True) -> tuple[Expr, Expr]:
        """
        A mostly naive separation of a Mul or Add into arguments that are not
        are dependent on deps. To obtain as complete a separation of variables
        as possible, use a separation method first, e.g.:

        * ``separatevars()`` to change Mul, Add and Pow (including exp) into Mul
        * ``.expand(mul=True)`` to change Add or Mul into Add
        * ``.expand(log=True)`` to change log expr into an Add

        The only non-naive thing that is done here is to respect noncommutative
        ordering of variables and to always return ``(0, 0)`` for ``self`` of
        zero regardless of hints.

        For nonzero ``self``, the returned tuple ``(i, d)`` has the following
        interpretation:

        * ``i`` has no variable that appears in deps
        * ``d`` will either have terms that contain variables that are in deps,
          or be equal to ``0`` (when ``self`` is an ``Add``) or ``1`` (when
          ``self`` is a ``Mul``)
        * if ``self`` is an Add then ``self = i + d``
        * if ``self`` is a Mul then ``self = i*d``
        * otherwise ``(self, S.One)`` or ``(S.One, self)`` is returned.

        To force the expression to be treated as an Add, use the argument
        ``as_Add=True``.

        The ``strict`` argument is deprecated and has no effect.

        Examples
        ========

        -- ``self`` is an Add

        >>> from sympy import sin, cos, exp
        >>> from sympy.abc import x, y, z

        >>> (x + x*y).as_independent(x)
        (0, x*y + x)
        >>> (x + x*y).as_independent(y)
        (x, x*y)
        >>> (2*x*sin(x) + y + x + z).as_independent(x)
        (y + z, 2*x*sin(x) + x)
        >>> (2*x*sin(x) + y + x + z).as_independent(x, y)
        (z, 2*x*sin(x) + x + y)

        -- ``self`` is a Mul

        >>> (x*sin(x)*cos(y)).as_independent(x)
        (cos(y), x*sin(x))

        Non-commutative terms cannot always be separated out when ``self`` is a
        Mul

        >>> from sympy import symbols
        >>> n1, n2, n3 = symbols('n1 n2 n3', commutative=False)
        >>> (n1 + n1*n2).as_independent(n2)
        (n1, n1*n2)
        >>> (n2*n1 + n1*n2).as_independent(n2)
        (0, n1*n2 + n2*n1)
        >>> (n1*n2*n3).as_independent(n1)
        (1, n1*n2*n3)
        >>> (n1*n2*n3).as_independent(n2)
        (n1, n2*n3)
        >>> ((x-n1)*(x-y)).as_independent(x)
        (1, (x - y)*(x - n1))

        -- ``self`` is anything else:

        >>> (sin(x)).as_independent(x)
        (1, sin(x))
        >>> (sin(x)).as_independent(y)
        (sin(x), 1)
        >>> exp(x+y).as_independent(x)
        (1, exp(x + y))

        -- force ``self`` to be treated as an Add:

        >>> (3*x).as_independent(x, as_Add=True)
        (0, 3*x)

        -- force ``self`` to be treated as a Mul:

        >>> (3+x).as_independent(x, as_Add=False)
        (1, x + 3)
        >>> (-3+x).as_independent(x, as_Add=False)
        (1, x - 3)

        Note how the below differs from the above in making the
        constant on the dep term positive.

        >>> (y*(-3+x)).as_independent(x)
        (y, x - 3)

        -- use ``.as_independent()`` for true independence testing instead of
           ``.has()``. The former considers only symbols in the free symbols
           while the latter considers all symbols

        >>> from sympy import Integral
        >>> I = Integral(x, (x, 1, 2))
        >>> I.has(x)
        True
        >>> x in I.free_symbols
        False
        >>> I.as_independent(x) == (I, 1)
        True
        >>> (I + x).as_independent(x) == (I, x)
        True

        Note: when trying to get independent terms, a separation method might
        need to be used first. In this case, it is important to keep track of
        what you send to this routine so you know how to interpret the returned
        values

        >>> from sympy import separatevars, log
        >>> separatevars(exp(x+y)).as_independent(x)
        (exp(y), exp(x))
        >>> (x + x*y).as_independent(y)
        (x, x*y)
        >>> separatevars(x + x*y).as_independent(y)
        (x, y + 1)
        >>> (x*(1 + y)).as_independent(y)
        (x, y + 1)
        >>> (x*(1 + y)).expand(mul=True).as_independent(y)
        (x, x*y)
        >>> a, b=symbols('a b', positive=True)
        >>> (log(a*b).expand(log=True)).as_independent(b)
        (log(a), log(b))

        See Also
        ========

        separatevars
        expand_log
        sympy.core.add.Add.as_two_terms
        sympy.core.mul.Mul.as_two_terms
        as_coeff_mul
        """
        from .symbol import Symbol
        from .add import _unevaluated_Add
        from .mul import _unevaluated_Mul
        if self is S.Zero:
            return (self, self)
        if as_Add is None:
            as_Add = self.is_Add
        syms, other = _sift_true_false(deps, lambda d: isinstance(d, Symbol))
        syms_set = set(syms)
        if other:

            def has(e):
                return e.has_xfree(syms_set) or e.has(*other)
        else:

            def has(e):
                return e.has_xfree(syms_set)
        if as_Add:
            if not self.is_Add:
                if has(self):
                    return (S.Zero, self)
                else:
                    return (self, S.Zero)
            depend, indep = _sift_true_false(self.args, has)
            return (self.func(*indep), _unevaluated_Add(*depend))
        else:
            if not self.is_Mul:
                if has(self):
                    return (S.One, self)
                else:
                    return (self, S.One)
            args, nc = self.args_cnc()
            depend, indep = _sift_true_false(args, has)
            for i, n in enumerate(nc):
                if has(n):
                    depend.extend(nc[i:])
                    break
                indep.append(n)
            return (self.func(*indep), _unevaluated_Mul(*depend))
    __round__ = round

    @property
    def func(self):
        """
        The top-level function in an expression.

        The following should hold for all objects::

            >> x == x.func(*x.args)

        Examples
        ========

        >>> from sympy.abc import x
        >>> a = 2*x
        >>> a.func
        <class 'sympy.core.mul.Mul'>
        >>> a.args
        (2, x)
        >>> a.func(*a.args)
        2*x
        >>> a == a.func(*a.args)
        True

        """
        return self.__class__

    @property
    def args(self) -> tuple[Basic, ...]:
        """Returns a tuple of arguments of 'self'.

        Examples
        ========

        >>> from sympy import cot
        >>> from sympy.abc import x, y

        >>> cot(x).args
        (x,)

        >>> cot(x).args[0]
        x

        >>> (x*y).args
        (x, y)

        >>> (x*y).args[1]
        y

        Notes
        =====

        Never use self._args, always use self.args.
        Only use _args in __new__ when creating a new function.
        Do not override .args() from Basic (so that it is easy to
        change the interface in the future if needed).
        """
        return self._args
