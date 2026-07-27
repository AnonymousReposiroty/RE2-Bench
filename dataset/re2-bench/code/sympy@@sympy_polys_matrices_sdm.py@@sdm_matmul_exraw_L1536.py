from collections import defaultdict

def sdm_matmul_exraw(A, B, K, m, o):
    #
    # Like sdm_matmul above except that:
    #
    # - Handles cases like 0*oo -> nan (sdm_matmul skips multiplication by zero)
    # - Uses K.sum (Add(*items)) for efficient addition of Expr
    #
    zero = K.zero
    C = {}
    B_knz = set(B)
    for i, Ai in A.items():
        Ci_list = defaultdict(list)
        Ai_knz = set(Ai)

        # Nonzero row/column pair
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            if zero * Aik == zero:
                # This is the main inner loop:
                for j, Bkj in B[k].items():
                    Ci_list[j].append(Aik * Bkj)
            else:
                for j in range(o):
                    Ci_list[j].append(Aik * B[k].get(j, zero))

        # Zero row in B, check for infinities in A
        for k in Ai_knz - B_knz:
            zAik = zero * Ai[k]
            if zAik != zero:
                for j in range(o):
                    Ci_list[j].append(zAik)

        # Add terms using K.sum (Add(*terms)) for efficiency
        Ci = {}
        for j, Cij_list in Ci_list.items():
            Cij = K.sum(Cij_list)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    # Find all infinities in B
    for k, Bk in B.items():
        for j, Bkj in Bk.items():
            if zero * Bkj != zero:
                for i in range(m):
                    Aik = A.get(i, {}).get(k, zero)
                    # If Aik is not zero then this was handled above
                    if Aik == zero:
                        Ci = C.get(i, {})
                        Cij = Ci.get(j, zero) + Aik * Bkj
                        if Cij != zero:
                            Ci[j] = Cij
                            C[i] = Ci
                        else:
                            Ci.pop(j, None)
                            if Ci:
                                C[i] = Ci
                            else:
                                C.pop(i, None)

    return C
