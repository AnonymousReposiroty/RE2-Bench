import numpy as np

def _broadcast_with_masks(*args, compress=False):
    """
    Broadcast inputs, combining all masked arrays.

    Parameters
    ----------
    *args : array-like
        The inputs to broadcast.
    compress : bool, default: False
        Whether to compress the masked arrays. If False, the masked values
        are replaced by NaNs.

    Returns
    -------
    list of array-like
        The broadcasted and masked inputs.
    """
    # extract the masks, if any
    masks = [k.mask for k in args if isinstance(k, np.ma.MaskedArray)]
    # broadcast to match the shape
    bcast = np.broadcast_arrays(*args, *masks)
    inputs = bcast[:len(args)]
    masks = bcast[len(args):]
    if masks:
        # combine the masks into one
        mask = np.logical_or.reduce(masks)
        # put mask on and compress
        if compress:
            inputs = [np.ma.array(k, mask=mask).compressed()
                      for k in inputs]
        else:
            inputs = [np.ma.array(k, mask=mask, dtype=float).filled(np.nan).ravel()
                      for k in inputs]
    else:
        inputs = [np.ravel(k) for k in inputs]
    return inputs
