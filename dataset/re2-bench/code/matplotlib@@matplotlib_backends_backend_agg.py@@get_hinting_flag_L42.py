import matplotlib as mpl
from matplotlib.ft2font import LoadFlags

def get_hinting_flag():
    mapping = {
        'default': LoadFlags.DEFAULT,
        'no_autohint': LoadFlags.NO_AUTOHINT,
        'force_autohint': LoadFlags.FORCE_AUTOHINT,
        'no_hinting': LoadFlags.NO_HINTING,
        True: LoadFlags.FORCE_AUTOHINT,
        False: LoadFlags.NO_HINTING,
        'either': LoadFlags.DEFAULT,
        'native': LoadFlags.NO_AUTOHINT,
        'auto': LoadFlags.FORCE_AUTOHINT,
        'none': LoadFlags.NO_HINTING,
    }
    return mapping[mpl.rcParams['text.hinting']]
