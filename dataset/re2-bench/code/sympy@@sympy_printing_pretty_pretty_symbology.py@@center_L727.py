def center(string, width, fillchar=' '):
    """Return a centered string of length determined by `line_width`
    that uses `fillchar` for padding.
    """
    left, right = center_pad(line_width(string), width, fillchar)
    return ''.join([left, string, right])
