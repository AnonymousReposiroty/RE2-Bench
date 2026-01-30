from types import SimpleNamespace
from .bezier import (
    NonIntersectingPathException, get_cos_sin, get_intersection,
    get_parallels, inside_circle, make_wedged_bezier2,
    split_bezier_intersecting_with_closedpath, split_path_inout)

class _Base:
    """
        A base class for connectionstyle classes. The subclass needs
        to implement a *connect* method whose call signature is::

          connect(posA, posB)

        where posA and posB are tuples of x, y coordinates to be
        connected.  The method needs to return a path connecting two
        points. This base class defines a __call__ method, and a few
        helper methods.
        """

    def _in_patch(self, patch):
        """
            Return a predicate function testing whether a point *xy* is
            contained in *patch*.
            """
        return lambda xy: patch.contains(SimpleNamespace(x=xy[0], y=xy[1]))[0]

    def _clip(self, path, in_start, in_stop):
        """
            Clip *path* at its start by the region where *in_start* returns
            True, and at its stop by the region where *in_stop* returns True.

            The original path is assumed to start in the *in_start* region and
            to stop in the *in_stop* region.
            """
        if in_start:
            try:
                _, path = split_path_inout(path, in_start)
            except ValueError:
                pass
        if in_stop:
            try:
                path, _ = split_path_inout(path, in_stop)
            except ValueError:
                pass
        return path

    def __call__(self, posA, posB, shrinkA=2.0, shrinkB=2.0, patchA=None, patchB=None):
        """
            Call the *connect* method to create a path between *posA* and
            *posB*; then clip and shrink the path.
            """
        path = self.connect(posA, posB)
        path = self._clip(path, self._in_patch(patchA) if patchA else None, self._in_patch(patchB) if patchB else None)
        path = self._clip(path, inside_circle(*path.vertices[0], shrinkA) if shrinkA else None, inside_circle(*path.vertices[-1], shrinkB) if shrinkB else None)
        return path
