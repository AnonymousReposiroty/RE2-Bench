import matplotlib as mpl

def _single_string_color_list(s, scalar_validator):
    """
    Convert the string *s* to a list of colors interpreting it either as a
    color sequence name, or a string containing single-letter colors.
    """
    try:
        colors = mpl.color_sequences[s]
    except KeyError:
        try:
            # Sometimes, a list of colors might be a single string
            # of single-letter colornames. So give that a shot.
            colors = [scalar_validator(v.strip()) for v in s if v.strip()]
        except ValueError:
            raise ValueError(f'{s!r} is neither a color sequence name nor can '
                             'it be interpreted as a list of colors')

    return colors
