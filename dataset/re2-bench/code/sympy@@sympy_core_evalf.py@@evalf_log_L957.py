from mpmath import (
    make_mpc, make_mpf, mp, mpc, mpf, nsum, quadts, quadosc, workprec)
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
                          fnan, finf, fninf, fnone, fone, fzero, mpf_abs, mpf_add,
                          mpf_atan, mpf_atan2, mpf_cmp, mpf_cos, mpf_e, mpf_exp, mpf_log, mpf_lt,
                          mpf_mul, mpf_neg, mpf_pi, mpf_pow, mpf_pow_int, mpf_shift, mpf_sin,
                          mpf_sqrt, normalize, round_nearest, to_int, to_str, mpf_tan)
from .singleton import S
from sympy.core.add import Add
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.complexes import Abs, re, im
from sympy.functions.elementary.complexes import re, im
from .add import Add
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log
from .add import Add
from .add import Add
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from .add import Add

def evalf_log(expr: 'log', prec: int, options: OPT_DICT) -> TMP_RES:
    if len(expr.args)>1:
        expr = expr.doit()
        return evalf(expr, prec, options)
    arg = expr.args[0]
    workprec = prec + 10
    result = evalf(arg, workprec, options)
    if result is S.ComplexInfinity:
        return result
    xre, xim, xacc, _ = result

    # evalf can return NoneTypes if chop=True
    # issue 18516, 19623
    if xre is xim is None:
        # Dear reviewer, I do not know what -inf is;
        # it looks to be (1, 0, -789, -3)
        # but I'm not sure in general,
        # so we just let mpmath figure
        # it out by taking log of 0 directly.
        # It would be better to return -inf instead.
        xre = fzero

    if xim:
        from sympy.functions.elementary.complexes import Abs
        from sympy.functions.elementary.exponential import log

        # XXX: use get_abs etc instead
        re = evalf_log(
            log(Abs(arg, evaluate=False), evaluate=False), prec, options)
        im = mpf_atan2(xim, xre or fzero, prec)
        return re[0], im, re[2], prec

    imaginary_term = (mpf_cmp(xre, fzero) < 0)

    re = mpf_log(mpf_abs(xre), prec, rnd)
    size = fastlog(re)
    if prec - size > workprec and re != fzero:
        from .add import Add
        # We actually need to compute 1+x accurately, not x
        add = Add(S.NegativeOne, arg, evaluate=False)
        xre, xim, _, _ = evalf_add(add, prec, options)
        prec2 = workprec - fastlog(xre)
        # xre is now x - 1 so we add 1 back here to calculate x
        re = mpf_log(mpf_abs(mpf_add(xre, fone, prec2)), prec, rnd)

    re_acc = prec

    if imaginary_term:
        return re, mpf_pi(prec), re_acc, prec
    else:
        return re, None, re_acc, None
