from mpmath.libmp import round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
    prec_to_dps, dps_to_prec)
from mpmath.libmp.backend import MPZ
from sympy.functions.elementary.complexes import sign

def mpf_norm(mpf, prec):
    """Return the mpf tuple normalized appropriately for the indicated
    precision after doing a check to see if zero should be returned or
    not when the mantissa is 0. ``mpf_normlize`` always assumes that this
    is zero, but it may not be since the mantissa for mpf's values "+inf",
    "-inf" and "nan" have a mantissa of zero, too.

    Note: this is not intended to validate a given mpf tuple, so sending
    mpf tuples that were not created by mpmath may produce bad results. This
    is only a wrapper to ``mpf_normalize`` which provides the check for non-
    zero mpfs that have a 0 for the mantissa.
    """
    sign, man, expt, bc = mpf
    if not man:
        # hack for mpf_normalize which does not do this;
        # it assumes that if man is zero the result is 0
        # (see issue 6639)
        if not bc:
            return fzero
        else:
            # don't change anything; this should already
            # be a well formed mpf tuple
            return mpf

    # Necessary if mpmath is using the gmpy backend
    from mpmath.libmp.backend import MPZ
    rv = mpf_normalize(sign, MPZ(man), expt, bc, prec, rnd)
    return rv
