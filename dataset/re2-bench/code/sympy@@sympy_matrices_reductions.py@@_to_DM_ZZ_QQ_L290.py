from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains import ZZ, QQ

def _to_DM_ZZ_QQ(M):
    # We have to test for _rep here because there are tests that otherwise fail
    # with e.g. "AttributeError: 'SubspaceOnlyMatrix' object has no attribute
    # '_rep'." There is almost certainly no value in such tests. The
    # presumption seems to be that someone could create a new class by
    # inheriting some of the Matrix classes and not the full set that is used
    # by the standard Matrix class but if anyone tried that it would fail in
    # many ways.
    if not hasattr(M, '_rep'):
        return None

    rep = M._rep
    K = rep.domain

    if K.is_ZZ:
        return rep
    elif K.is_QQ:
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep
    else:
        if not all(e.is_Rational for e in M):
            return None
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep.convert_to(QQ)
