import matplotlib as mpl

def get_epoch():
    """
    Get the epoch used by `.dates`.

    Returns
    -------
    epoch : str
        String for the epoch (parsable by `numpy.datetime64`).
    """
    global _epoch

    _epoch = mpl._val_or_rc(_epoch, 'date.epoch')
    return _epoch
