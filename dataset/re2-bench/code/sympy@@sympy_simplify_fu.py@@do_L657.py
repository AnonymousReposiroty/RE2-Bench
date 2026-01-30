from collections import defaultdict
from sympy.core.add import Add
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)

def f(rv):
    if not rv.is_Add:
        return rv

    def do(rv, first=True):
        # args which can be expressed as A*(cos(a)*cos(b)+/-sin(a)*sin(b))
        # or B*(cos(a)*sin(b)+/-cos(b)*sin(a)) can be combined into
        # A*f(a+/-b) where f is either sin or cos.
        #
        # If there are more than two args, the pairs which "work" will have
        # a gcd extractable and the remaining two terms will have the above
        # structure -- all pairs must be checked to find the ones that
        # work.

        if not rv.is_Add:
            return rv

        args = list(ordered(rv.args))
        if len(args) != 2:
            hit = False
            for i in range(len(args)):
                ai = args[i]
                if ai is None:
                    continue
                for j in range(i + 1, len(args)):
                    aj = args[j]
                    if aj is None:
                        continue
                    was = ai + aj
                    new = do(was)
                    if new != was:
                        args[i] = new  # update in place
                        args[j] = None
                        hit = True
                        break  # go to next i
            if hit:
                rv = Add(*[_f for _f in args if _f])
                if rv.is_Add:
                    rv = do(rv)

            return rv

        # two-arg Add
        split = trig_split(*args, two=True)
        if not split:
            return rv
        gcd, n1, n2, a, b, same = split

        # identify and get c1 to be cos then apply rule if possible
        if same:  # coscos, sinsin
            gcd = n1*gcd
            if n1 == n2:
                return gcd*cos(a - b)
            return gcd*cos(a + b)
        else:  #cossin, cossin
            gcd = n1*gcd
            if n1 == n2:
                return gcd*sin(a + b)
            return gcd*sin(b - a)

    rv = process_common_addends(
        rv, do, lambda x: tuple(ordered(x.free_symbols)))

    # need to check for inducible pairs in ratio of sqrt(3):1 that
    # appeared in different lists when sorting by coefficient
    while rv.is_Add:
        byrad = defaultdict(list)
        for a in rv.args:
            hit = 0
            if a.is_Mul:
                for ai in a.args:
                    if ai.is_Pow and ai.exp is S.Half and \
                            ai.base.is_Integer:
                        byrad[ai].append(a)
                        hit = 1
                        break
            if not hit:
                byrad[S.One].append(a)

        # no need to check all pairs -- just check for the onees
        # that have the right ratio
        args = []
        for a in byrad:
            for b in [_ROOT3()*a, _invROOT3()]:
                if b in byrad:
                    for i in range(len(byrad[a])):
                        if byrad[a][i] is None:
                            continue
                        for j in range(len(byrad[b])):
                            if byrad[b][j] is None:
                                continue
                            was = Add(byrad[a][i] + byrad[b][j])
                            new = do(was)
                            if new != was:
                                args.append(new)
                                byrad[a][i] = None
                                byrad[b][j] = None
                                break
        if args:
            rv = Add(*(args + [Add(*[_f for _f in v if _f])
                for v in byrad.values()]))
        else:
            rv = do(rv)  # final pass to resolve any new inducible pairs
            break

    return rv
