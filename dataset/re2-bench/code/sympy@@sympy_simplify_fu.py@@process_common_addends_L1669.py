from collections import defaultdict
from sympy.core.add import Add
from sympy.core.singleton import S

def process_common_addends(rv, do, key2=None, key1=True):
    """Apply ``do`` to addends of ``rv`` that (if ``key1=True``) share at least
    a common absolute value of their coefficient and the value of ``key2`` when
    applied to the argument. If ``key1`` is False ``key2`` must be supplied and
    will be the only key applied.
    """

    # collect by absolute value of coefficient and key2
    absc = defaultdict(list)
    if key1:
        for a in rv.args:
            c, a = a.as_coeff_Mul()
            if c < 0:
                c = -c
                a = -a  # put the sign on `a`
            absc[(c, key2(a) if key2 else 1)].append(a)
    elif key2:
        for a in rv.args:
            absc[(S.One, key2(a))].append(a)
    else:
        raise ValueError('must have at least one key')

    args = []
    hit = False
    for k in absc:
        v = absc[k]
        c, _ = k
        if len(v) > 1:
            e = Add(*v, evaluate=False)
            new = do(e)
            if new != e:
                e = new
                hit = True
            args.append(c*e)
        else:
            args.append(c*v[0])
    if hit:
        rv = Add(*args)

    return rv
