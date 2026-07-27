from mpmath import (
    make_mpc, make_mpf, mp, mpc, mpf, nsum, quadts, quadosc, workprec)
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
                          fnan, finf, fninf, fnone, fone, fzero, mpf_abs, mpf_add,
                          mpf_atan, mpf_atan2, mpf_cmp, mpf_cos, mpf_e, mpf_exp, mpf_log, mpf_lt,
                          mpf_mul, mpf_neg, mpf_pi, mpf_pow, mpf_pow_int, mpf_shift, mpf_sin,
                          mpf_sqrt, normalize, round_nearest, to_int, to_str, mpf_tan)
from .singleton import S
from sympy.functions.elementary.complexes import Abs, re, im
from sympy.functions.elementary.complexes import re, im
from sympy.functions.elementary.complexes import Abs, im, re

def quad_to_mpmath(q, ctx=None):
    """Turn the quad returned by ``evalf`` into an ``mpf`` or ``mpc``. """
    mpc = make_mpc if ctx is None else ctx.make_mpc
    mpf = make_mpf if ctx is None else ctx.make_mpf
    if q is S.ComplexInfinity:
        raise NotImplementedError
    re, im, _, _ = q
    if im:
        if not re:
            re = fzero
        return mpc((re, im))
    elif re:
        return mpf(re)
    else:
        return mpf(fzero)
