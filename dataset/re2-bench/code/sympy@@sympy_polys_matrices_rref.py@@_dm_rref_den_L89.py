def _dm_rref_den(M, *, keep_domain=True, method='auto'):
    """
    Compute the reduced row echelon form of a ``DomainMatrix`` with denominator.

    This function is the implementation of :meth:`DomainMatrix.rref_den`.

    Chooses the best algorithm depending on the domain, shape, and sparsity of
    the matrix as well as things like the bit count in the case of :ref:`ZZ` or
    :ref:`QQ`. The result is returned over the same domain as the input matrix
    unless ``keep_domain=False`` in which case the result might be over an
    associated ring or field domain.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        The ``DomainMatrix`` method that calls this function.
    sympy.polys.matrices.rref._dm_rref
        Alternative function for computing RREF without denominator.
    """
    method, use_fmt = _dm_rref_choose_method(M, method, denominator=True)

    M, old_fmt = _dm_to_fmt(M, use_fmt)

    if method == 'FF':
        # Use fraction-free GJ over the current domain.
        M_rref, den, pivots = _dm_rref_den_FF(M)

    elif method == 'GJ':
        # Use Gauss-Jordan with division over the associated field.
        M_rref_f, pivots = _dm_rref_GJ(_to_field(M))

        # Convert back to the ring?
        if keep_domain and M_rref_f.domain != M.domain:
            _, M_rref = M_rref_f.clear_denoms(convert=True)

            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
        else:
            # Possibly an associated field
            M_rref = M_rref_f
            den = M_rref.domain.one

    elif method == 'CD':
        # Clear denominators and use fraction-free GJ in the associated ring.
        _, Mr = M.clear_denoms_rowwise(convert=True)

        M_rref_r, den, pivots = _dm_rref_den_FF(Mr)

        if keep_domain and M_rref_r.domain != M.domain:
            # Convert back to the field
            M_rref = _to_field(M_rref_r) / den
            den = M.domain.one
        else:
            # Possibly an associated ring
            M_rref = M_rref_r

            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
    else:
        raise ValueError(f"Unknown method for rref: {method}")

    M_rref, _ = _dm_to_fmt(M_rref, old_fmt)

    # Invariants:
    #   - M_rref is in the same format (sparse or dense) as the input matrix.
    #   - If keep_domain=True then M_rref and den are in the same domain as the
    #     input matrix
    #   - If keep_domain=False then M_rref might be in an associated ring or
    #     field domain but den is always in the same domain as M_rref.

    return M_rref, den, pivots
