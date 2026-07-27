import numpy as np

def contiguous_regions(mask):
    """
    Return a list of (ind0, ind1) such that ``mask[ind0:ind1].all()`` is
    True and we cover all such regions.
    """
    mask = np.asarray(mask, dtype=bool)

    if not mask.size:
        return []

    # Find the indices of region changes, and correct offset
    idx, = np.nonzero(mask[:-1] != mask[1:])
    idx += 1

    # List operations are faster for moderately sized arrays
    idx = idx.tolist()

    # Add first and/or last index if needed
    if mask[0]:
        idx = [0] + idx
    if mask[-1]:
        idx.append(len(mask))

    return list(zip(idx[::2], idx[1::2]))
