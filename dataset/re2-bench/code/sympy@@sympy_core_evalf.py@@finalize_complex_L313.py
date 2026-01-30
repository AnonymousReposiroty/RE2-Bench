from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
                          fnan, finf, fninf, fnone, fone, fzero, mpf_abs, mpf_add,
                          mpf_atan, mpf_atan2, mpf_cmp, mpf_cos, mpf_e, mpf_exp, mpf_log, mpf_lt,
                          mpf_mul, mpf_neg, mpf_pi, mpf_pow, mpf_pow_int, mpf_shift, mpf_sin,
                          mpf_sqrt, normalize, round_nearest, to_int, to_str, mpf_tan)
from sympy.functions.elementary.complexes import Abs, re, im
from sympy.functions.elementary.complexes import re, im
from sympy.functions.elementary.complexes import Abs, im, re

def finalize_complex(re: MPF_TUP, im: MPF_TUP, prec: int) -> TMP_RES:
    if re == fzero and im == fzero:
        raise ValueError("got complex zero with unknown accuracy")
    elif re == fzero:
        return None, im, None, prec
    elif im == fzero:
        return re, None, prec, None

    size_re = fastlog(re)
    size_im = fastlog(im)
    if size_re > size_im:
        re_acc = prec
        im_acc = prec + min(-(size_re - size_im), 0)
    else:
        im_acc = prec
        re_acc = prec + min(-(size_im - size_re), 0)
    return re, im, re_acc, im_acc
