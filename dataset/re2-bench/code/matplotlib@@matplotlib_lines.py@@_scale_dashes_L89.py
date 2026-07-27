import matplotlib as mpl

def _scale_dashes(offset, dashes, lw):
    if not mpl.rcParams['lines.scale_dashes']:
        return offset, dashes
    scaled_offset = offset * lw
    scaled_dashes = ([x * lw if x is not None else None for x in dashes]
                     if dashes is not None else None)
    return scaled_offset, scaled_dashes
