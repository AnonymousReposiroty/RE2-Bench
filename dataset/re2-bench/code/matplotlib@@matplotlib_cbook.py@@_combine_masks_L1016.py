import numpy as np
from numpy.exceptions import VisibleDeprecationWarning  # numpy >= 1.25
from numpy import VisibleDeprecationWarning

def _combine_masks(*args):
    """
    Find all masked and/or non-finite points in a set of arguments,
    and return the arguments as masked arrays with a common mask.

    Arguments can be in any of 5 categories:

    1) 1-D masked arrays
    2) 1-D ndarrays
    3) ndarrays with more than one dimension
    4) other non-string iterables
    5) anything else

    The first argument must be in one of the first four categories;
    any argument with a length differing from that of the first
    argument (and hence anything in category 5) then will be
    passed through unchanged.

    Masks are obtained from all arguments of the correct length
    in categories 1, 2, and 4; a point is bad if masked in a masked
    array or if it is a nan or inf.  No attempt is made to
    extract a mask from categories 2 and 4 if `numpy.isfinite`
    does not yield a Boolean array.  Category 3 is included to
    support RGB or RGBA ndarrays, which are assumed to have only
    valid values and which are passed through unchanged.

    All input arguments that are not passed unchanged are returned
    as masked arrays if any masked points are found, otherwise as
    ndarrays.

    """
    if not len(args):
        return ()
    if is_scalar_or_string(args[0]):
        raise ValueError("First argument must be a sequence")
    nrecs = len(args[0])
    margs = []  # Output args; some may be modified.
    seqlist = [False] * len(args)  # Flags: True if output will be masked.
    masks = []    # List of masks.
    for i, x in enumerate(args):
        if is_scalar_or_string(x) or len(x) != nrecs:
            margs.append(x)  # Leave it unmodified.
        else:
            if isinstance(x, np.ma.MaskedArray) and x.ndim > 1:
                raise ValueError("Masked arrays must be 1-D")
            try:
                x = np.asanyarray(x)
            except (VisibleDeprecationWarning, ValueError):
                # NumPy 1.19 raises a warning about ragged arrays, but we want
                # to accept basically anything here.
                x = np.asanyarray(x, dtype=object)
            if x.ndim == 1:
                x = safe_masked_invalid(x)
                seqlist[i] = True
                if np.ma.is_masked(x):
                    masks.append(np.ma.getmaskarray(x))
            margs.append(x)  # Possibly modified.
    if len(masks):
        mask = np.logical_or.reduce(masks)
        for i, x in enumerate(margs):
            if seqlist[i]:
                margs[i] = np.ma.array(x, mask=mask)
    return margs
