def _dm_rref_choose_method(M, method, *, denominator=False):
    """Choose the fastest method for computing RREF for M."""

    if method != 'auto':
        if method.endswith('_dense'):
            method = method[:-len('_dense')]
            use_fmt = 'dense'
        else:
            use_fmt = 'sparse'

    else:
        # The sparse implementations are always faster
        use_fmt = 'sparse'

        K = M.domain

        if K.is_ZZ:
            method = _dm_rref_choose_method_ZZ(M, denominator=denominator)
        elif K.is_QQ:
            method = _dm_rref_choose_method_QQ(M, denominator=denominator)
        elif K.is_RR or K.is_CC:
            # TODO: Add partial pivot support to the sparse implementations.
            method = 'GJ'
            use_fmt = 'dense'
        elif K.is_EX and M.rep.fmt == 'dense' and not denominator:
            # Do not switch to the sparse implementation for EX because the
            # domain does not have proper canonicalization and the sparse
            # implementation gives equivalent but non-identical results over EX
            # from performing arithmetic in a different order. Specifically
            # test_issue_23718 ends up getting a more complicated expression
            # when using the sparse implementation. Probably the best fix for
            # this is something else but for now we stick with the dense
            # implementation for EX if the matrix is already dense.
            method = 'GJ'
            use_fmt = 'dense'
        else:
            # This is definitely suboptimal. More work is needed to determine
            # the best method for computing RREF over different domains.
            if denominator:
                method = 'FF'
            else:
                method = 'GJ'

    return method, use_fmt
