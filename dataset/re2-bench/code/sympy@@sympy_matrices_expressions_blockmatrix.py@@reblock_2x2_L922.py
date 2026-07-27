def reblock_2x2(expr):
    """
    Reblock a BlockMatrix so that it has 2x2 blocks of block matrices.  If
    possible in such a way that the matrix continues to be invertible using the
    classical 2x2 block inversion formulas.
    """
    if not isinstance(expr, BlockMatrix) or not all(d > 2 for d in expr.blockshape):
        return expr

    BM = BlockMatrix  # for brevity's sake
    rowblocks, colblocks = expr.blockshape
    blocks = expr.blocks
    for i in range(1, rowblocks):
        for j in range(1, colblocks):
            # try to split rows at i and cols at j
            A = bc_unpack(BM(blocks[:i, :j]))
            B = bc_unpack(BM(blocks[:i, j:]))
            C = bc_unpack(BM(blocks[i:, :j]))
            D = bc_unpack(BM(blocks[i:, j:]))

            formula = _choose_2x2_inversion_formula(A, B, C, D)
            if formula is not None:
                return BlockMatrix([[A, B], [C, D]])

    # else: nothing worked, just split upper left corner
    return BM([[blocks[0, 0], BM(blocks[0, 1:])],
               [BM(blocks[1:, 0]), BM(blocks[1:, 1:])]])
