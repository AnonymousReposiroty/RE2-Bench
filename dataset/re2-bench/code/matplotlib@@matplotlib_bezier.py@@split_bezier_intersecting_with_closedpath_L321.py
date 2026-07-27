def split_bezier_intersecting_with_closedpath(
        bezier, inside_closedpath, tolerance=0.01):
    """
    Split a Bézier curve into two at the intersection with a closed path.

    Parameters
    ----------
    bezier : (N, 2) array-like
        Control points of the Bézier segment. See `.BezierSegment`.
    inside_closedpath : callable
        A function returning True if a given point (x, y) is inside the
        closed path. See also `.find_bezier_t_intersecting_with_closedpath`.
    tolerance : float
        The tolerance for the intersection. See also
        `.find_bezier_t_intersecting_with_closedpath`.

    Returns
    -------
    left, right
        Lists of control points for the two Bézier segments.
    """

    bz = BezierSegment(bezier)

    t0, t1 = find_bezier_t_intersecting_with_closedpath(
        lambda t: tuple(bz(t)), inside_closedpath, tolerance=tolerance)

    _left, _right = split_de_casteljau(bezier, (t0 + t1) / 2.)
    return _left, _right
