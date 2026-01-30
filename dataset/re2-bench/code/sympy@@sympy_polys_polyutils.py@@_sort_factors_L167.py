def _sort_factors(factors, **args):
    """Sort low-level factors in increasing 'complexity' order. """

    # XXX: GF(p) does not support comparisons so we need a key function to sort
    # the factors if python-flint is being used. A better solution might be to
    # add a sort key method to each domain.
    def order_key(factor):
        if isinstance(factor, _GF_types):
            return int(factor)
        elif isinstance(factor, list):
            return [order_key(f) for f in factor]
        else:
            return factor

    def order_if_multiple_key(factor):
        (f, n) = factor
        return (len(f), n, order_key(f))

    def order_no_multiple_key(f):
        return (len(f), order_key(f))

    if args.get('multiple', True):
        return sorted(factors, key=order_if_multiple_key)
    else:
        return sorted(factors, key=order_no_multiple_key)
