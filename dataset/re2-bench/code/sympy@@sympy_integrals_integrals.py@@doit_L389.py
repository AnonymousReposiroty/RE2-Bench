from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import oo, pi
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.functions import Piecewise, sqrt, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.special.singularity_functions import Heaviside
from .rationaltools import ratint
from sympy.matrices import MatrixBase
from sympy.polys import Poly, PolynomialError
from sympy.series.formal import FormalPowerSeries
from sympy.utilities.misc import filldedent
from .deltafunctions import deltaintegrate
from .meijerint import meijerint_definite, meijerint_indefinite, _debug
from .trigonometry import trigintegrate
from sympy.concrete.summations import Sum
from sympy.integrals.risch import risch_integrate, NonElementaryIntegral
from sympy.integrals.manualintegrate import manualintegrate
from sympy.simplify.fu import sincos_to_sum
from sympy.concrete.summations import Sum
from .singularityfunctions import singularityintegrate
from sympy.integrals.heurisch import (heurisch as heurisch_,
                                      heurisch_wrapper)

class Integral(AddWithLimits):
    """Represents unevaluated integral."""
    __slots__ = ()
    args: tuple[Expr, Tuple]

    def doit(self, **hints):
        """
        Perform the integration using any hints given.

        Examples
        ========

        >>> from sympy import Piecewise, S
        >>> from sympy.abc import x, t
        >>> p = x**2 + Piecewise((0, x/t < 0), (1, True))
        >>> p.integrate((t, S(4)/5, 1), (x, -1, 1))
        1/3

        See Also
        ========

        sympy.integrals.trigonometry.trigintegrate
        sympy.integrals.heurisch.heurisch
        sympy.integrals.rationaltools.ratint
        as_sum : Approximate the integral using a sum
        """
        if not hints.get('integrals', True):
            return self
        deep = hints.get('deep', True)
        meijerg = hints.get('meijerg', None)
        conds = hints.get('conds', 'piecewise')
        risch = hints.get('risch', None)
        heurisch = hints.get('heurisch', None)
        manual = hints.get('manual', None)
        if len(list(filter(None, (manual, meijerg, risch, heurisch)))) > 1:
            raise ValueError('At most one of manual, meijerg, risch, heurisch can be True')
        elif manual:
            meijerg = risch = heurisch = False
        elif meijerg:
            manual = risch = heurisch = False
        elif risch:
            manual = meijerg = heurisch = False
        elif heurisch:
            manual = meijerg = risch = False
        eval_kwargs = {'meijerg': meijerg, 'risch': risch, 'manual': manual, 'heurisch': heurisch, 'conds': conds}
        if conds not in ('separate', 'piecewise', 'none'):
            raise ValueError('conds must be one of "separate", "piecewise", "none", got: %s' % conds)
        if risch and any((len(xab) > 1 for xab in self.limits)):
            raise ValueError('risch=True is only allowed for indefinite integrals.')
        if self.is_zero:
            return S.Zero
        from sympy.concrete.summations import Sum
        if isinstance(self.function, Sum):
            if any((v in self.function.limits[0] for v in self.variables)):
                raise ValueError('Limit of the sum cannot be an integration variable.')
            if any((l.is_infinite for l in self.function.limits[0][1:])):
                return self
            _i = self
            _sum = self.function
            return _sum.func(_i.func(_sum.function, *_i.limits).doit(), *_sum.limits).doit()
        function = self.function
        function = function.replace(lambda x: isinstance(x, Heaviside) and x.args[1] * 2 != 1, lambda x: Heaviside(x.args[0]))
        if deep:
            function = function.doit(**hints)
        if function.is_zero:
            return S.Zero
        if isinstance(function, MatrixBase):
            return function.applyfunc(lambda f: self.func(f, *self.limits).doit(**hints))
        if isinstance(function, FormalPowerSeries):
            if len(self.limits) > 1:
                raise NotImplementedError
            xab = self.limits[0]
            if len(xab) > 1:
                return function.integrate(xab, **eval_kwargs)
            else:
                return function.integrate(xab[0], **eval_kwargs)
        reps = {}
        for xab in self.limits:
            if len(xab) != 3:
                continue
            x, a, b = xab
            l = (a, b)
            if all((i.is_nonnegative for i in l)) and (not x.is_nonnegative):
                d = Dummy(positive=True)
            elif all((i.is_nonpositive for i in l)) and (not x.is_nonpositive):
                d = Dummy(negative=True)
            elif all((i.is_real for i in l)) and (not x.is_real):
                d = Dummy(real=True)
            else:
                d = None
            if d:
                reps[x] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did
        undone_limits = []
        ulj = set()
        for xab in self.limits:
            if len(xab) == 1:
                uli = set(xab[:1])
            elif len(xab) == 2:
                uli = xab[1].free_symbols
            elif len(xab) == 3:
                uli = xab[1].free_symbols.union(xab[2].free_symbols)
            if xab[0] in ulj or any((v[0] in uli for v in undone_limits)):
                undone_limits.append(xab)
                ulj.update(uli)
                function = self.func(*[function] + [xab])
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            if function.has(Abs, sign) and (len(xab) < 3 and all((x.is_extended_real for x in xab)) or (len(xab) == 3 and all((x.is_extended_real and (not x.is_infinite) for x in xab[1:])))):
                xr = Dummy('xr', real=True)
                function = function.xreplace({xab[0]: xr}).rewrite(Piecewise).xreplace({xr: xab[0]})
            elif function.has(Min, Max):
                function = function.rewrite(Piecewise)
            if function.has(Piecewise) and (not isinstance(function, Piecewise)):
                function = piecewise_fold(function)
            if isinstance(function, Piecewise):
                if len(xab) == 1:
                    antideriv = function._eval_integral(xab[0], **eval_kwargs)
                else:
                    antideriv = self._eval_integral(function, xab[0], **eval_kwargs)
            else:

                def try_meijerg(function, xab):
                    ret = None
                    if len(xab) == 3 and meijerg is not False:
                        x, a, b = xab
                        try:
                            res = meijerint_definite(function, x, a, b)
                        except NotImplementedError:
                            _debug('NotImplementedError from meijerint_definite')
                            res = None
                        if res is not None:
                            f, cond = res
                            if conds == 'piecewise':
                                u = self.func(function, (x, a, b))
                                return Piecewise((f, cond), (u, True), evaluate=False)
                            elif conds == 'separate':
                                if len(self.limits) != 1:
                                    raise ValueError(filldedent('\n                                        conds=separate not supported in\n                                        multiple integrals'))
                                ret = (f, cond)
                            else:
                                ret = f
                    return ret
                meijerg1 = meijerg
                if meijerg is not False and len(xab) == 3 and xab[1].is_extended_real and xab[2].is_extended_real and (not function.is_Poly) and (xab[1].has(oo, -oo) or xab[2].has(oo, -oo)):
                    ret = try_meijerg(function, xab)
                    if ret is not None:
                        function = ret
                        continue
                    meijerg1 = False
                if meijerg1 is False and meijerg is True:
                    antideriv = None
                else:
                    antideriv = self._eval_integral(function, xab[0], **eval_kwargs)
                    if antideriv is None and meijerg is True:
                        ret = try_meijerg(function, xab)
                        if ret is not None:
                            function = ret
                            continue
            final = hints.get('final', True)
            if final and (not isinstance(antideriv, Integral)) and (antideriv is not None):
                for atan_term in antideriv.atoms(atan):
                    atan_arg = atan_term.args[0]
                    for tan_part in atan_arg.atoms(tan):
                        x1 = Dummy('x1')
                        tan_exp1 = atan_arg.subs(tan_part, x1)
                        coeff = tan_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = tan_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term, sign(coeff) * pi * floor((a - pi / 2) / pi)))
                    for cot_part in atan_arg.atoms(cot):
                        x1 = Dummy('x1')
                        cot_exp1 = atan_arg.subs(cot_part, x1)
                        coeff = cot_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = cot_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term, sign(coeff) * pi * floor(a / pi)))
            if antideriv is None:
                undone_limits.append(xab)
                function = self.func(*[function] + [xab]).factor()
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            elif len(xab) == 1:
                function = antideriv
            else:
                if len(xab) == 3:
                    x, a, b = xab
                elif len(xab) == 2:
                    x, b = xab
                    a = None
                else:
                    raise NotImplementedError
                if deep:
                    if isinstance(a, Basic):
                        a = a.doit(**hints)
                    if isinstance(b, Basic):
                        b = b.doit(**hints)
                if antideriv.is_Poly:
                    gens = list(antideriv.gens)
                    gens.remove(x)
                    antideriv = antideriv.as_expr()
                    function = antideriv._eval_interval(x, a, b)
                    function = Poly(function, *gens)
                else:

                    def is_indef_int(g, x):
                        return isinstance(g, Integral) and any((i == (x,) for i in g.limits))

                    def eval_factored(f, x, a, b):
                        args = []
                        for g in Mul.make_args(f):
                            if is_indef_int(g, x):
                                args.append(g._eval_interval(x, a, b))
                            else:
                                args.append(g)
                        return Mul(*args)
                    integrals, others, piecewises = ([], [], [])
                    for f in Add.make_args(antideriv):
                        if any((is_indef_int(g, x) for g in Mul.make_args(f))):
                            integrals.append(f)
                        elif any((isinstance(g, Piecewise) for g in Mul.make_args(f))):
                            piecewises.append(piecewise_fold(f))
                        else:
                            others.append(f)
                    uneval = Add(*[eval_factored(f, x, a, b) for f in integrals])
                    try:
                        evalued = Add(*others)._eval_interval(x, a, b)
                        evalued_pw = piecewise_fold(Add(*piecewises))._eval_interval(x, a, b)
                        function = uneval + evalued + evalued_pw
                    except NotImplementedError:
                        undone_limits.append(xab)
                        function = self.func(*[function] + [xab])
                        factored_function = function.factor()
                        if not isinstance(factored_function, Integral):
                            function = factored_function
        return function

    def _eval_integral(self, f, x, meijerg=None, risch=None, manual=None, heurisch=None, conds='piecewise', final=None):
        """
        Calculate the anti-derivative to the function f(x).

        Explanation
        ===========

        The following algorithms are applied (roughly in this order):

        1. Simple heuristics (based on pattern matching and integral table):

           - most frequently used functions (e.g. polynomials, products of
             trig functions)

        2. Integration of rational functions:

           - A complete algorithm for integrating rational functions is
             implemented (the Lazard-Rioboo-Trager algorithm).  The algorithm
             also uses the partial fraction decomposition algorithm
             implemented in apart() as a preprocessor to make this process
             faster.  Note that the integral of a rational function is always
             elementary, but in general, it may include a RootSum.

        3. Full Risch algorithm:

           - The Risch algorithm is a complete decision
             procedure for integrating elementary functions, which means that
             given any elementary function, it will either compute an
             elementary antiderivative, or else prove that none exists.
             Currently, part of transcendental case is implemented, meaning
             elementary integrals containing exponentials, logarithms, and
             (soon!) trigonometric functions can be computed.  The algebraic
             case, e.g., functions containing roots, is much more difficult
             and is not implemented yet.

           - If the routine fails (because the integrand is not elementary, or
             because a case is not implemented yet), it continues on to the
             next algorithms below.  If the routine proves that the integrals
             is nonelementary, it still moves on to the algorithms below,
             because we might be able to find a closed-form solution in terms
             of special functions.  If risch=True, however, it will stop here.

        4. The Meijer G-Function algorithm:

           - This algorithm works by first rewriting the integrand in terms of
             very general Meijer G-Function (meijerg in SymPy), integrating
             it, and then rewriting the result back, if possible.  This
             algorithm is particularly powerful for definite integrals (which
             is actually part of a different method of Integral), since it can
             compute closed-form solutions of definite integrals even when no
             closed-form indefinite integral exists.  But it also is capable
             of computing many indefinite integrals as well.

           - Another advantage of this method is that it can use some results
             about the Meijer G-Function to give a result in terms of a
             Piecewise expression, which allows to express conditionally
             convergent integrals.

           - Setting meijerg=True will cause integrate() to use only this
             method.

        5. The "manual integration" algorithm:

           - This algorithm tries to mimic how a person would find an
             antiderivative by hand, for example by looking for a
             substitution or applying integration by parts. This algorithm
             does not handle as many integrands but can return results in a
             more familiar form.

           - Sometimes this algorithm can evaluate parts of an integral; in
             this case integrate() will try to evaluate the rest of the
             integrand using the other methods here.

           - Setting manual=True will cause integrate() to use only this
             method.

        6. The Heuristic Risch algorithm:

           - This is a heuristic version of the Risch algorithm, meaning that
             it is not deterministic.  This is tried as a last resort because
             it can be very slow.  It is still used because not enough of the
             full Risch algorithm is implemented, so that there are still some
             integrals that can only be computed using this method.  The goal
             is to implement enough of the Risch and Meijer G-function methods
             so that this can be deleted.

             Setting heurisch=True will cause integrate() to use only this
             method. Set heurisch=False to not use it.

        """
        from sympy.integrals.risch import risch_integrate, NonElementaryIntegral
        from sympy.integrals.manualintegrate import manualintegrate
        if risch:
            try:
                return risch_integrate(f, x, conds=conds)
            except NotImplementedError:
                return None
        if manual:
            try:
                result = manualintegrate(f, x)
                if result is not None and result.func != Integral:
                    return result
            except (ValueError, PolynomialError):
                pass
        eval_kwargs = {'meijerg': meijerg, 'risch': risch, 'manual': manual, 'heurisch': heurisch, 'conds': conds}
        if isinstance(f, Poly) and (not (manual or meijerg or risch)):
            return f.integrate(x)
        if isinstance(f, Piecewise):
            return f.piecewise_integrate(x, **eval_kwargs)
        if not f.has(x):
            return f * x
        poly = f.as_poly(x)
        if poly is not None and (not (manual or meijerg or risch)):
            return poly.integrate().as_expr()
        if risch is not False:
            try:
                result, i = risch_integrate(f, x, separate_integral=True, conds=conds)
            except NotImplementedError:
                pass
            else:
                if i:
                    if result == 0:
                        return NonElementaryIntegral(f, x).doit(risch=False)
                    else:
                        return result + i.doit(risch=False)
                else:
                    return result
        from sympy.simplify.fu import sincos_to_sum
        parts = []
        args = Add.make_args(f)
        for g in args:
            coeff, g = g.as_independent(x)
            if g is S.One and (not meijerg):
                parts.append(coeff * x)
                continue
            order_term = g.getO()
            if order_term is not None:
                h = self._eval_integral(g.removeO(), x, **eval_kwargs)
                if h is not None:
                    h_order_expr = self._eval_integral(order_term.expr, x, **eval_kwargs)
                    if h_order_expr is not None:
                        h_order_term = order_term.func(h_order_expr, *order_term.variables)
                        parts.append(coeff * (h + h_order_term))
                        continue
                return None
            if g.is_Pow and (not g.exp.has(x)) and (not meijerg):
                a = Wild('a', exclude=[x])
                b = Wild('b', exclude=[x])
                M = g.base.match(a * x + b)
                if M is not None:
                    if g.exp == -1:
                        h = log(g.base)
                    elif conds != 'piecewise':
                        h = g.base ** (g.exp + 1) / (g.exp + 1)
                    else:
                        h1 = log(g.base)
                        h2 = g.base ** (g.exp + 1) / (g.exp + 1)
                        h = Piecewise((h2, Ne(g.exp, -1)), (h1, True))
                    parts.append(coeff * h / M[a])
                    continue
            if g.is_rational_function(x) and (not (manual or meijerg or risch)):
                parts.append(coeff * ratint(g, x))
                continue
            if not (manual or meijerg or risch):
                h = trigintegrate(g, x, conds=conds)
                if h is not None:
                    parts.append(coeff * h)
                    continue
                h = deltaintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue
                from .singularityfunctions import singularityintegrate
                h = singularityintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue
                if risch is not False:
                    try:
                        h, i = risch_integrate(g, x, separate_integral=True, conds=conds)
                    except NotImplementedError:
                        h = None
                    else:
                        if i:
                            h = h + i.doit(risch=False)
                        parts.append(coeff * h)
                        continue
                if heurisch is not False:
                    from sympy.integrals.heurisch import heurisch as heurisch_, heurisch_wrapper
                    try:
                        if conds == 'piecewise':
                            h = heurisch_wrapper(g, x, hints=[])
                        else:
                            h = heurisch_(g, x, hints=[])
                    except PolynomialError:
                        h = None
            else:
                h = None
            if meijerg is not False and h is None:
                try:
                    h = meijerint_indefinite(g, x)
                except NotImplementedError:
                    _debug('NotImplementedError from meijerint_definite')
                if h is not None:
                    parts.append(coeff * h)
                    continue
            if h is None and manual is not False:
                try:
                    result = manualintegrate(g, x)
                    if result is not None and (not isinstance(result, Integral)):
                        if result.has(Integral) and (not manual):
                            new_eval_kwargs = eval_kwargs
                            new_eval_kwargs['manual'] = False
                            new_eval_kwargs['final'] = False
                            result = result.func(*[arg.doit(**new_eval_kwargs) if arg.has(Integral) else arg for arg in result.args]).expand(multinomial=False, log=False, power_exp=False, power_base=False)
                        if not result.has(Integral):
                            parts.append(coeff * result)
                            continue
                except (ValueError, PolynomialError):
                    pass
            if not h and len(args) == 1:
                f = sincos_to_sum(f).expand(mul=True, deep=False)
                if f.is_Add:
                    return self._eval_integral(f, x, **eval_kwargs)
            if h is not None:
                parts.append(coeff * h)
            else:
                return None
        return Add(*parts)
