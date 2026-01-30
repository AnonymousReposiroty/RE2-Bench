def _dm_to_fmt(M, fmt):
    """Convert a matrix to the given format and return the old format."""
    old_fmt = M.rep.fmt
    if old_fmt == fmt:
        pass
    elif fmt == 'dense':
        M = M.to_dense()
    elif fmt == 'sparse':
        M = M.to_sparse()
    else:
        raise ValueError(f'Unknown format: {fmt}') # pragma: no cover
    return M, old_fmt
