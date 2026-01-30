def _dm_rref_den_FF(M):
    """Compute RREF using fraction-free Gauss-Jordan elimination."""
    if M.rep.fmt == 'sparse':
        return _dm_rref_den_FF_sparse(M)
    else:
        return _dm_rref_den_FF_dense(M)
