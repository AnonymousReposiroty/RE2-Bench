from typing import TYPE_CHECKING, ClassVar, overload, Literal
from collections import defaultdict
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.matrices.expressions import MatrixExpr

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

    @classmethod
    def flatten(cls, seq):
        """Return commutative, noncommutative and order arguments by
        combining related terms.

        Notes
        =====
            * In an expression like ``a*b*c``, Python process this through SymPy
              as ``Mul(Mul(a, b), c)``. This can have undesirable consequences.

              -  Sometimes terms are not combined as one would like:
                 {c.f. https://github.com/sympy/sympy/issues/4596}

                >>> from sympy import Mul, sqrt
                >>> from sympy.abc import x, y, z
                >>> 2*(x + 1) # this is the 2-arg Mul behavior
                2*x + 2
                >>> y*(x + 1)*2
                2*y*(x + 1)
                >>> 2*(x + 1)*y # 2-arg result will be obtained first
                y*(2*x + 2)
                >>> Mul(2, x + 1, y) # all 3 args simultaneously processed
                2*y*(x + 1)
                >>> 2*((x + 1)*y) # parentheses can control this behavior
                2*y*(x + 1)

                Powers with compound bases may not find a single base to
                combine with unless all arguments are processed at once.
                Post-processing may be necessary in such cases.
                {c.f. https://github.com/sympy/sympy/issues/5728}

                >>> a = sqrt(x*sqrt(y))
                >>> a**3
                (x*sqrt(y))**(3/2)
                >>> Mul(a,a,a)
                (x*sqrt(y))**(3/2)
                >>> a*a*a
                x*sqrt(y)*sqrt(x*sqrt(y))
                >>> _.subs(a.base, z).subs(z, a.base)
                (x*sqrt(y))**(3/2)

              -  If more than two terms are being multiplied then all the
                 previous terms will be re-processed for each new argument.
                 So if each of ``a``, ``b`` and ``c`` were :class:`Mul`
                 expression, then ``a*b*c`` (or building up the product
                 with ``*=``) will process all the arguments of ``a`` and
                 ``b`` twice: once when ``a*b`` is computed and again when
                 ``c`` is multiplied.

                 Using ``Mul(a, b, c)`` will process all arguments once.

            * The results of Mul are cached according to arguments, so flatten
              will only be called once for ``Mul(a, b, c)``. If you can
              structure a calculation so the arguments are most likely to be
              repeats then this can save time in computing the answer. For
              example, say you had a Mul, M, that you wished to divide by ``d[i]``
              and multiply by ``n[i]`` and you suspect there are many repeats
              in ``n``. It would be better to compute ``M*n[i]/d[i]`` rather
              than ``M/d[i]*n[i]`` since every time n[i] is a repeat, the
              product, ``M*n[i]`` will be returned without flattening -- the
              cached value will be returned. If you divide by the ``d[i]``
              first (and those are more unique than the ``n[i]``) then that will
              create a new Mul, ``M/d[i]`` the args of which will be traversed
              again when it is multiplied by ``n[i]``.

              {c.f. https://github.com/sympy/sympy/issues/5706}

              This consideration is moot if the cache is turned off.

            NB
            --
              The validity of the above notes depends on the implementation
              details of Mul and flatten which may change at any time. Therefore,
              you should only consider them when your code is highly performance
              sensitive.

              Removal of 1 from the sequence is already handled by AssocOp.__new__.
        """
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.matrices.expressions import MatrixExpr
        rv = None
        if len(seq) == 2:
            a, b = seq
            if b.is_Rational:
                a, b = (b, a)
                seq = [a, b]
            assert a is not S.One
            if a.is_Rational and (not a.is_zero):
                r, b = b.as_coeff_Mul()
                if b.is_Add:
                    if r is not S.One:
                        ar = a * r
                        if ar is S.One:
                            arb = b
                        else:
                            arb = cls(a * r, b, evaluate=False)
                        rv = ([arb], [], None)
                    elif global_parameters.distribute and b.is_commutative:
                        newb = Add(*[_keep_coeff(a, bi) for bi in b.args])
                        rv = ([newb], [], None)
            if rv:
                return rv
        c_part = []
        nc_part = []
        nc_seq = []
        coeff = S.One
        c_powers = []
        num_exp = []
        neg1e = S.Zero
        pnum_rat = {}
        order_symbols = None
        for o in seq:
            if o.is_Order:
                o, order_symbols = o.as_expr_variables(order_symbols)
            if o.is_Mul:
                if o.is_commutative:
                    seq.extend(o.args)
                else:
                    for q in o.args:
                        if q.is_commutative:
                            seq.append(q)
                        else:
                            nc_seq.append(q)
                    seq.append(NC_Marker)
                continue
            elif o.is_Number:
                if o is S.NaN or (coeff is S.ComplexInfinity and o.is_zero):
                    return ([S.NaN], [], None)
                elif coeff.is_Number or isinstance(coeff, AccumBounds):
                    coeff *= o
                    if coeff is S.NaN:
                        return ([S.NaN], [], None)
                continue
            elif isinstance(o, AccumBounds):
                coeff = o.__mul__(coeff)
                continue
            elif o is S.ComplexInfinity:
                if not coeff:
                    return ([S.NaN], [], None)
                coeff = S.ComplexInfinity
                continue
            elif not coeff and isinstance(o, Add) and any((_ in (S.NegativeInfinity, S.ComplexInfinity, S.Infinity) for __ in o.args for _ in Mul.make_args(__))):
                return ([S.NaN], [], None)
            elif o is S.ImaginaryUnit:
                neg1e += S.Half
                continue
            elif o.is_commutative:
                b, e = o.as_base_exp()
                if o.is_Pow:
                    if b.is_Number:
                        if e.is_Rational:
                            if e.is_Integer:
                                coeff *= Pow(b, e)
                                continue
                            elif e.is_negative:
                                seq.append(Pow(b, e))
                                continue
                            elif b.is_negative:
                                neg1e += e
                                b = -b
                            if b is not S.One:
                                pnum_rat.setdefault(b, []).append(e)
                            continue
                        elif b.is_positive or e.is_integer:
                            num_exp.append((b, e))
                            continue
                c_powers.append((b, e))
            else:
                if o is not NC_Marker:
                    nc_seq.append(o)
                while nc_seq:
                    o = nc_seq.pop(0)
                    if not nc_part:
                        nc_part.append(o)
                        continue
                    o1 = nc_part.pop()
                    b1, e1 = o1.as_base_exp()
                    b2, e2 = o.as_base_exp()
                    new_exp = e1 + e2
                    if b1 == b2 and (not new_exp.is_Add):
                        o12 = b1 ** new_exp
                        if o12.is_commutative:
                            seq.append(o12)
                            continue
                        else:
                            nc_seq.insert(0, o12)
                    else:
                        nc_part.extend([o1, o])

        def _gather(c_powers):
            common_b = {}
            for b, e in c_powers:
                co = e.as_coeff_Mul()
                common_b.setdefault(b, {}).setdefault(co[1], []).append(co[0])
            for b, d in common_b.items():
                for di, li in d.items():
                    d[di] = Add(*li)
            new_c_powers = []
            for b, e in common_b.items():
                new_c_powers.extend([(b, c * t) for t, c in e.items()])
            return new_c_powers
        c_powers = _gather(c_powers)
        num_exp = _gather(num_exp)
        for i in range(2):
            new_c_powers = []
            changed = False
            for b, e in c_powers:
                if e.is_zero:
                    if (b.is_Add or b.is_Mul) and any((infty in b.args for infty in (S.ComplexInfinity, S.Infinity, S.NegativeInfinity))):
                        return ([S.NaN], [], None)
                    continue
                if e is S.One:
                    if b.is_Number:
                        coeff *= b
                        continue
                    p = b
                if e is not S.One:
                    p = Pow(b, e)
                    if p.is_Pow and (not b.is_Pow):
                        bi = b
                        b, e = p.as_base_exp()
                        if b != bi:
                            changed = True
                c_part.append(p)
                new_c_powers.append((b, e))
            if changed and len({b for b, e in new_c_powers}) != len(new_c_powers):
                c_part = []
                c_powers = _gather(new_c_powers)
            else:
                break
        inv_exp_dict = {}
        for b, e in num_exp:
            inv_exp_dict.setdefault(e, []).append(b)
        for e, b in inv_exp_dict.items():
            inv_exp_dict[e] = cls(*b)
        c_part.extend([Pow(b, e) for e, b in inv_exp_dict.items() if e])
        comb_e = {}
        for b, e in pnum_rat.items():
            comb_e.setdefault(Add(*e), []).append(b)
        del pnum_rat
        num_rat = []
        for e, b in comb_e.items():
            b = cls(*b)
            if e.q == 1:
                coeff *= Pow(b, e)
                continue
            if e.p > e.q:
                e_i, ep = divmod(e.p, e.q)
                coeff *= Pow(b, e_i)
                e = Rational(ep, e.q)
            num_rat.append((b, e))
        del comb_e
        pnew = defaultdict(list)
        i = 0
        while i < len(num_rat):
            bi, ei = num_rat[i]
            if bi == 1:
                i += 1
                continue
            grow = []
            for j in range(i + 1, len(num_rat)):
                bj, ej = num_rat[j]
                g = bi.gcd(bj)
                if g is not S.One:
                    e = ei + ej
                    if e.q == 1:
                        coeff *= Pow(g, e)
                    else:
                        if e.p > e.q:
                            e_i, ep = divmod(e.p, e.q)
                            coeff *= Pow(g, e_i)
                            e = Rational(ep, e.q)
                        grow.append((g, e))
                    num_rat[j] = (bj / g, ej)
                    bi = bi / g
                    if bi is S.One:
                        break
            if bi is not S.One:
                obj = Pow(bi, ei)
                if obj.is_Number:
                    coeff *= obj
                else:
                    for obj in Mul.make_args(obj):
                        if obj.is_Number:
                            coeff *= obj
                        else:
                            assert obj.is_Pow
                            bi, ei = obj.args
                            pnew[ei].append(bi)
            num_rat.extend(grow)
            i += 1
        for e, b in pnew.items():
            pnew[e] = cls(*b)
        if neg1e:
            p, q = neg1e.as_numer_denom()
            n, p = divmod(p, q)
            if n % 2:
                coeff = -coeff
            if q == 2:
                c_part.append(S.ImaginaryUnit)
            elif p:
                neg1e = Rational(p, q)
                for e, b in pnew.items():
                    if e == neg1e and b.is_positive:
                        pnew[e] = -b
                        break
                else:
                    c_part.append(Pow(S.NegativeOne, neg1e, evaluate=False))
        c_part.extend([Pow(b, e) for e, b in pnew.items()])
        if coeff in (S.Infinity, S.NegativeInfinity):

            def _handle_for_oo(c_part, coeff_sign):
                new_c_part = []
                for t in c_part:
                    if t.is_extended_positive:
                        continue
                    if t.is_extended_negative:
                        coeff_sign *= -1
                        continue
                    new_c_part.append(t)
                return (new_c_part, coeff_sign)
            c_part, coeff_sign = _handle_for_oo(c_part, 1)
            nc_part, coeff_sign = _handle_for_oo(nc_part, coeff_sign)
            coeff *= coeff_sign
        if coeff is S.ComplexInfinity:
            c_part = [c for c in c_part if not (fuzzy_not(c.is_zero) and c.is_extended_real is not None)]
            nc_part = [c for c in nc_part if not (fuzzy_not(c.is_zero) and c.is_extended_real is not None)]
        elif coeff.is_zero:
            if any((isinstance(c, MatrixExpr) for c in nc_part)):
                return ([coeff], nc_part, order_symbols)
            if any((c.is_finite == False for c in c_part)):
                return ([S.NaN], [], order_symbols)
            return ([coeff], [], order_symbols)
        _new = []
        for i in c_part:
            if i.is_Number:
                coeff *= i
            else:
                _new.append(i)
        c_part = _new
        _mulsort(c_part)
        if coeff is not S.One:
            c_part.insert(0, coeff)
        if global_parameters.distribute and (not nc_part) and (len(c_part) == 2) and c_part[0].is_Number and c_part[0].is_finite and c_part[1].is_Add:
            coeff = c_part[0]
            c_part = [Add(*[coeff * f for f in c_part[1].args])]
        return (c_part, nc_part, order_symbols)
    _eval_is_commutative = lambda self: _fuzzy_group((a.is_commutative for a in self.args))
