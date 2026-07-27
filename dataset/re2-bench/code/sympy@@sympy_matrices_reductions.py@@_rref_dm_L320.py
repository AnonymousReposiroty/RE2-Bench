def _rref_dm(dM):
    """Compute the reduced row echelon form of a DomainMatrix."""
    K = dM.domain

    if K.is_ZZ:
        dM_rref, den, pivots = dM.rref_den(keep_domain=False)
        dM_rref = dM_rref.to_field() / den
    elif K.is_QQ:
        dM_rref, pivots = dM.rref()
    else:
        assert False  # pragma: no cover

    M_rref = dM_rref.to_Matrix()

    return M_rref, pivots
