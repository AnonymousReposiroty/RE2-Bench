from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.functions import Piecewise, sqrt, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from .rationaltools import ratint
from sympy.polys import Poly, PolynomialError
from .deltafunctions import deltaintegrate
from .meijerint import meijerint_definite, meijerint_indefinite, _debug
from .trigonometry import trigintegrate
from sympy.integrals.risch import risch_integrate, NonElementaryIntegral
from sympy.integrals.manualintegrate import manualintegrate
from sympy.simplify.fu import sincos_to_sum
from .singularityfunctions import singularityintegrate
from sympy.integrals.heurisch import (heurisch as heurisch_,
                                      heurisch_wrapper)

class Integral(AddWithLimits):
    """Represents unevaluated integral."""
    __slots__ = ()
    args: tuple[Expr, Tuple]

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
