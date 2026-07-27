from sympy.core.containers import Tuple

def normalize(i, parentsize):
    if isinstance(i, slice):
        i = (i.start, i.stop, i.step)
    if not isinstance(i, (tuple, list, Tuple)):
        if (i < 0) == True:
            i += parentsize
        i = (i, i+1, 1)
    i = list(i)
    if len(i) == 2:
        i.append(1)
    start, stop, step = i
    start = start or 0
    if stop is None:
        stop = parentsize
    if (start < 0) == True:
        start += parentsize
    if (stop < 0) == True:
        stop += parentsize
    step = step or 1

    if ((stop - start) * step < 1) == True:
        raise IndexError()

    return (start, stop, step)
