from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
               lines as mlines, transforms)
from .bezier import (
    NonIntersectingPathException, get_cos_sin, get_intersection,
    get_parallels, inside_circle, make_wedged_bezier2,
    split_bezier_intersecting_with_closedpath, split_path_inout)
from .path import Path
from matplotlib.bezier import get_normal_points

class _Curve(_Base):
    """
        A simple arrow which will work with any path instance. The
        returned path is the concatenation of the original path, and at
        most two paths representing the arrow head or bracket at the start
        point and at the end point. The arrow heads can be either open
        or closed.
        """
    arrow = '-'
    fillbegin = fillend = False

    def __init__(self, head_length=0.4, head_width=0.2, widthA=1.0, widthB=1.0, lengthA=0.2, lengthB=0.2, angleA=0, angleB=0, scaleA=None, scaleB=None):
        """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head, relative to *mutation_size*.
            head_width : float, default: 0.2
                Width of the arrow head, relative to *mutation_size*.
            widthA, widthB : float, default: 1.0
                Width of the bracket.
            lengthA, lengthB : float, default: 0.2
                Length of the bracket.
            angleA, angleB : float, default: 0
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            scaleA, scaleB : float, default: *mutation_size*
                The scale of the brackets.
            """
        self.head_length, self.head_width = (head_length, head_width)
        self.widthA, self.widthB = (widthA, widthB)
        self.lengthA, self.lengthB = (lengthA, lengthB)
        self.angleA, self.angleB = (angleA, angleB)
        self.scaleA, self.scaleB = (scaleA, scaleB)
        self._beginarrow_head = False
        self._beginarrow_bracket = False
        self._endarrow_head = False
        self._endarrow_bracket = False
        if '-' not in self.arrow:
            raise ValueError("arrow must have the '-' between the two heads")
        beginarrow, endarrow = self.arrow.split('-', 1)
        if beginarrow == '<':
            self._beginarrow_head = True
            self._beginarrow_bracket = False
        elif beginarrow == '<|':
            self._beginarrow_head = True
            self._beginarrow_bracket = False
            self.fillbegin = True
        elif beginarrow in (']', '|'):
            self._beginarrow_head = False
            self._beginarrow_bracket = True
        if endarrow == '>':
            self._endarrow_head = True
            self._endarrow_bracket = False
        elif endarrow == '|>':
            self._endarrow_head = True
            self._endarrow_bracket = False
            self.fillend = True
        elif endarrow in ('[', '|'):
            self._endarrow_head = False
            self._endarrow_bracket = True
        super().__init__()

    def _get_bracket(self, x0, y0, x1, y1, width, length, angle):
        cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)
        from matplotlib.bezier import get_normal_points
        x1, y1, x2, y2 = get_normal_points(x0, y0, cos_t, sin_t, width)
        dx, dy = (length * cos_t, length * sin_t)
        vertices_arrow = [(x1 + dx, y1 + dy), (x1, y1), (x2, y2), (x2 + dx, y2 + dy)]
        codes_arrow = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
        if angle:
            trans = transforms.Affine2D().rotate_deg_around(x0, y0, angle)
            vertices_arrow = trans.transform(vertices_arrow)
        return (vertices_arrow, codes_arrow)
