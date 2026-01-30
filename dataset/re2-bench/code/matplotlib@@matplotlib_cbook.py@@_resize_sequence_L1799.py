import itertools

def _resize_sequence(seq, N):
    """
    Trim the given sequence to exactly N elements.

    If there are more elements in the sequence, cut it.
    If there are less elements in the sequence, repeat them.

    Implementation detail: We maintain type stability for the output for
    N <= len(seq). We simply return a list for N > len(seq); this was good
    enough for the present use cases but is not a fixed design decision.
    """
    num_elements = len(seq)
    if N == num_elements:
        return seq
    elif N < num_elements:
        return seq[:N]
    else:
        return list(itertools.islice(itertools.cycle(seq), N))
