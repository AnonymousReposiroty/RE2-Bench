def validate_fontstretch(s):
    stretchvalues = [
        'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed',
        'normal', 'semi-expanded', 'expanded', 'extra-expanded',
        'ultra-expanded']
    # Note: Historically, stretchvalues have been case-sensitive in Matplotlib
    if s in stretchvalues:
        return s
    try:
        return int(s)
    except (ValueError, TypeError) as e:
        raise ValueError(f'{s} is not a valid font stretch.') from e
