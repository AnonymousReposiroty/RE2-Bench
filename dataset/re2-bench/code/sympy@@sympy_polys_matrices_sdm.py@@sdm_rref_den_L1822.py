def sdm_rref_den(A, K):
    """
    Return the reduced row echelon form (RREF) of A with denominator.

    The RREF is computed using fraction-free Gauss-Jordan elimination.

    Explanation
    ===========

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ in [1]_. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic. This implementation is also
    optimized for sparse matrices.

    The domain $K$ must support exact division (``K.exquo``) but does not need
    to be a field. This method is suitable for most exact rings and fields like
    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or
    :ref:`K(x)` it might be more efficient to clear denominators and use
    :ref:`ZZ` or :ref:`K[x]` instead.

    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.

    Examples
    ========

    >>> from sympy.polys.matrices.sdm import sdm_rref_den
    >>> from sympy.polys.domains import ZZ
    >>> A = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}
    >>> A_rref, den, pivots = sdm_rref_den(A, ZZ)
    >>> A_rref
    {0: {0: -2}, 1: {1: -2}}
    >>> den
    -2
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        Higher-level interface to ``sdm_rref_den`` that would usually be used
        instead of calling this function directly.
    sympy.polys.matrices.sdm.sdm_rref_den
        The ``SDM`` method that uses this function.
    sdm_irref
        Computes RREF using field division.
    ddm_irref_den
        The dense version of this algorithm.

    References
    ==========

    .. [1] Fraction-free algorithms for linear and polynomial equations.
        George C. Nakos , Peter R. Turner , Robert M. Williams.
        https://dl.acm.org/doi/10.1145/271130.271133
    """
    #
    # We represent each row of the matrix as a dict mapping column indices to
    # nonzero elements. We will build the RREF matrix starting from an empty
    # matrix and appending one row at a time. At each step we will have the
    # RREF of the rows we have processed so far.
    #
    # Our representation of the RREF divides it into three parts:
    #
    # 1. Fully reduced rows having only a single nonzero element (the pivot).
    # 2. Partially reduced rows having nonzeros after the pivot.
    # 3. The current denominator and divisor.
    #
    # For example if the incremental RREF might be:
    #
    #   [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #   [0, 0, 2, 0, 0, 0, 7, 0, 0, 0]
    #   [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    #   [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    #   [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
    #
    # Here the second row is partially reduced and the other rows are fully
    # reduced. The denominator would be 2 in this case. We distinguish the
    # fully reduced rows because we can handle them more efficiently when
    # adding a new row.
    #
    # When adding a new row we need to multiply it by the current denominator.
    # Then we reduce the new row by cross cancellation with the previous rows.
    # Then if it is not reduced to zero we take its leading entry as the new
    # pivot, cross cancel the new row from the previous rows and update the
    # denominator. In the fraction-free version this last step requires
    # multiplying and dividing the whole matrix by the new pivot and the
    # current divisor. The advantage of building the RREF one row at a time is
    # that in the sparse case we only need to work with the relatively sparse
    # upper rows of the matrix. The simple version of FFGJ in [1] would
    # multiply and divide all the dense lower rows at each step.

    # Handle the trivial cases.
    if not A:
        return ({}, K.one, [])
    elif len(A) == 1:
        Ai, = A.values()
        j = min(Ai)
        Aij = Ai[j]
        return ({0: Ai.copy()}, Aij, [j])

    # For inexact domains like RR[x] we use quo and discard the remainder.
    # Maybe it would be better for K.exquo to do this automatically.
    if K.is_Exact:
        exquo = K.exquo
    else:
        exquo = K.quo

    # Make sure we have the rows in order to make this deterministic from the
    # outset.
    _, rows_in_order = zip(*sorted(A.items()))

    col_to_row_reduced = {}
    col_to_row_unreduced = {}
    reduced = col_to_row_reduced.keys()
    unreduced = col_to_row_unreduced.keys()

    # Our representation of the RREF so far.
    A_rref_rows = []
    denom = None
    divisor = None

    # The rows that remain to be added to the RREF. These are sorted by the
    # column index of their leading entry. Note that sorted() is stable so the
    # previous sort by unique row index is still needed to make this
    # deterministic (there may be multiple rows with the same leading column).
    A_rows = sorted(rows_in_order, key=min)

    for Ai in A_rows:

        # All fully reduced columns can be immediately discarded.
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced}

        # We need to multiply the new row by the current denominator to bring
        # it into the same scale as the previous rows and then cross-cancel to
        # reduce it wrt the previous unreduced rows. All pivots in the previous
        # rows are equal to denom so the coefficients we need to make a linear
        # combination of the previous rows to cancel into the new row are just
        # the ones that are already in the new row *before* we multiply by
        # denom. We compute that linear combination first and then multiply the
        # new row by denom before subtraction.
        Ai_cancel = {}

        for j in unreduced & Ai.keys():
            # Remove the pivot column from the new row since it would become
            # zero anyway.
            Aij = Ai.pop(j)

            Aj = A_rref_rows[col_to_row_unreduced[j]]

            for k, Ajk in Aj.items():
                Aik_cancel = Ai_cancel.get(k)
                if Aik_cancel is None:
                    Ai_cancel[k] = Aij * Ajk
                else:
                    Aik_cancel = Aik_cancel + Aij * Ajk
                    if Aik_cancel:
                        Ai_cancel[k] = Aik_cancel
                    else:
                        Ai_cancel.pop(k)

        # Multiply the new row by the current denominator and subtract.
        Ai_nz = set(Ai)
        Ai_cancel_nz = set(Ai_cancel)

        d = denom or K.one

        for k in Ai_cancel_nz - Ai_nz:
            Ai[k] = -Ai_cancel[k]

        for k in Ai_nz - Ai_cancel_nz:
            Ai[k] = Ai[k] * d

        for k in Ai_cancel_nz & Ai_nz:
            Aik = Ai[k] * d - Ai_cancel[k]
            if Aik:
                Ai[k] = Aik
            else:
                Ai.pop(k)

        # Now Ai has the same scale as the other rows and is reduced wrt the
        # unreduced rows.

        # If the row is reduced to zero then discard it.
        if not Ai:
            continue

        # Choose a pivot for this row.
        j = min(Ai)
        Aij = Ai.pop(j)

        # Cross cancel the unreduced rows by the new row.
        #     a[k][l] = (a[i][j]*a[k][l] - a[k][j]*a[i][l]) / divisor
        for pk, k in list(col_to_row_unreduced.items()):

            Ak = A_rref_rows[k]

            if j not in Ak:
                # This row is already reduced wrt the new row but we need to
                # bring it to the same scale as the new denominator. This step
                # is not needed in sdm_irref.
                for l, Akl in Ak.items():
                    Akl = Akl * Aij
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                continue

            Akj = Ak.pop(j)
            Ai_nz = set(Ai)
            Ak_nz = set(Ak)

            for l in Ai_nz - Ak_nz:
                Ak[l] = - Akj * Ai[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            # This loop also not needed in sdm_irref.
            for l in Ak_nz - Ai_nz:
                Ak[l] = Aij * Ak[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            for l in Ai_nz & Ak_nz:
                Akl = Aij * Ak[l] - Akj * Ai[l]
                if Akl:
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                else:
                    Ak.pop(l)

            if not Ak:
                col_to_row_unreduced.pop(pk)
                col_to_row_reduced[pk] = k

        i = len(A_rref_rows)
        A_rref_rows.append(Ai)
        if Ai:
            col_to_row_unreduced[j] = i
        else:
            col_to_row_reduced[j] = i

        # Update the denominator.
        if not K.is_one(Aij):
            if denom is None:
                denom = Aij
            else:
                denom *= Aij

        if divisor is not None:
            denom = exquo(denom, divisor)

        # Update the divisor.
        divisor = denom

    if denom is None:
        denom = K.one

    # Sort the rows by their leading column index.
    col_to_row = {**col_to_row_reduced, **col_to_row_unreduced}
    row_to_col = {i: j for j, i in col_to_row.items()}
    A_rref_rows_col = [(row_to_col[i], Ai) for i, Ai in enumerate(A_rref_rows)]
    pivots, A_rref = zip(*sorted(A_rref_rows_col))
    pivots = list(pivots)

    # Insert the pivot values
    for i, Ai in enumerate(A_rref):
        Ai[pivots[i]] = denom

    A_rref_sdm = dict(enumerate(A_rref))

    return A_rref_sdm, denom, pivots
