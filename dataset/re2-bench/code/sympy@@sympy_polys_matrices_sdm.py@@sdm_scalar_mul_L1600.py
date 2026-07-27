def sdm_scalar_mul(A, b, op, K):
    """
    Handles special cases like 0 * oo -> nan by creating a dense result
    when necessary. For all other cases, it uses the fast sparse approach.
    """

    zero = K.zero
    if K.is_EXRAW and op(zero, b) != zero:
        Csdm = sdm_scalar_mul_exraw(A, b, op, K)
    else:
        Csdm = unop_dict(A, lambda aij: op(aij, b))
    return Csdm
