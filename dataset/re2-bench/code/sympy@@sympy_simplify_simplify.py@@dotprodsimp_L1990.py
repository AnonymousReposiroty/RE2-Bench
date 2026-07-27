from sympy.core import (Basic, S, Add, Mul, Pow, Symbol, sympify,
                        expand_func, Function, Dummy, Expr, factor_terms,
                        expand_power_exp, Eq)
from sympy.core.numbers import Float, I, pi, Rational, equal_valued
from sympy.polys import together, cancel, factor
from sympy.simplify.radsimp import radsimp, fraction, collect_abs

def dotprodsimp(expr: Expr, withsimp: bool = False) -> Expr | tuple[Expr, bool]:
    """Simplification for a sum of products targeted at the kind of blowup that
    occurs during summation of products. Intended to reduce expression blowup
    during matrix multiplication or other similar operations. Only works with
    algebraic expressions and does not recurse into non.

    Parameters
    ==========

    withsimp : bool, optional
        Specifies whether a flag should be returned along with the expression
        to indicate roughly whether simplification was successful. It is used
        in ``MatrixArithmetic._eval_pow_by_recursion`` to avoid attempting to
        simplify an expression repetitively which does not simplify.
    """

    def count_ops_alg(expr: Expr) -> tuple[int, bool]:
        """Optimized count algebraic operations with no recursion into
        non-algebraic args that ``core.function.count_ops`` does. Also returns
        whether rational functions may be present according to negative
        exponents of powers or non-number fractions.

        Returns
        =======

        ops, ratfunc : int, bool
            ``ops`` is the number of algebraic operations starting at the top
            level expression (not recursing into non-alg children). ``ratfunc``
            specifies whether the expression MAY contain rational functions
            which ``cancel`` MIGHT optimize.
        """

        ops = 0
        args: list[Basic] = [expr]
        ratfunc = False

        while args:
            a = args.pop()

            if not isinstance(a, Basic):
                continue

            if isinstance(a, Rational):
                if a is not S.One: # -1/3 = NEG + DIV
                    ops += bool (a.p < 0) + bool (a.q != 1)

            elif isinstance(a, Mul):
                if a.could_extract_minus_sign():
                    ops += 1
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a

                n, d = fraction(a)

                if n.is_Integer:
                    ops += 1 + bool (n < 0)
                    args.append(d) # won't be -Mul but could be Add

                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                        ratfunc=True

                    ops += 1
                    args.append(n) # could be -Mul

                else:
                    ops += len(a.args) - 1
                    args.extend(a.args)

            elif isinstance(a, Add):
                laargs = len(a.args)
                negs   = 0

                for ai in a.args:
                    if ai.could_extract_minus_sign():
                        negs += 1
                        ai    = -ai
                    args.append(ai)

                ops += laargs - (negs != laargs) # -x - y = NEG + SUB

            elif isinstance(a, Pow):
                ops += 1
                args.append(a.base)

                if not ratfunc:
                    ratfunc = a.exp.is_negative is not False

        return ops, ratfunc

    def nonalg_subs_dummies(expr: Expr, dummies: dict[Expr, Dummy]) -> Expr:
        """Substitute dummy variables for non-algebraic expressions to avoid
        evaluation of non-algebraic terms that ``polys.polytools.cancel`` does.
        """

        if not expr.args:
            return expr

        if isinstance(expr, (Add, Mul, Pow)):
            args = None

            for i, a in enumerate(expr.args):
                c = nonalg_subs_dummies(a, dummies) # type: ignore

                if c is a:
                    continue

                if args is None:
                    args = list(expr.args)

                args[i] = c

            if args is None:
                return expr

            return expr.func(*args)

        return dummies.setdefault(expr, Dummy())

    simplified = False # doesn't really mean simplified, rather "can simplify again"

    if isinstance(expr, Basic) and (expr.is_Add or expr.is_Mul or expr.is_Pow):
        expr2 = expr.expand(deep=True, modulus=None, power_base=False,
            power_exp=False, mul=True, log=False, multinomial=True, basic=False)

        if expr2 != expr:
            expr       = expr2
            simplified = True

        exprops, ratfunc = count_ops_alg(expr)

        if exprops >= 6: # empirically tested cutoff for expensive simplification
            if ratfunc:
                dummies: dict[Expr, Dummy] = {}
                expr2   = nonalg_subs_dummies(expr, dummies)

                if expr2 is expr or count_ops_alg(expr2)[0] >= 6: # check again after substitution
                    expr3 = cancel(expr2)

                    if expr3 != expr2:
                        expr       = expr3.subs([(d, e) for e, d in dummies.items()])
                        simplified = True

        # very special case: x/(x-1) - 1/(x-1) -> 1
        elif (exprops == 5 and expr.is_Add and expr.args [0].is_Mul and
                expr.args [1].is_Mul and expr.args [0].args [-1].is_Pow and
                expr.args [1].args [-1].is_Pow and
              expr.args [0].args [-1].exp is S.NegativeOne and # type: ignore
              expr.args [1].args [-1].exp is S.NegativeOne): # type: ignore

            expr2    = together (expr)
            expr2ops = count_ops_alg(expr2)[0]

            if expr2ops < exprops:
                expr       = expr2
                simplified = True

        else:
            simplified = True

    return (expr, simplified) if withsimp else expr
