import numpy as np
from .path import Path

class _Base:
    """
        Arrow Transmuter Base class

        ArrowTransmuterBase and its derivatives are used to make a fancy
        arrow around a given path. The __call__ method returns a path
        (which will be used to create a PathPatch instance) and a boolean
        value indicating the path is open therefore is not fillable.  This
        class is not an artist and actual drawing of the fancy arrow is
        done by the FancyArrowPatch class.
        """

    def transmute(self, path, mutation_size, linewidth):
        """
            The transmute method is the very core of the ArrowStyle class and
            must be overridden in the subclasses. It receives the *path*
            object along which the arrow will be drawn, and the
            *mutation_size*, with which the arrow head etc. will be scaled.
            The *linewidth* may be used to adjust the path so that it does not
            pass beyond the given points. It returns a tuple of a `.Path`
            instance and a boolean. The boolean value indicate whether the
            path can be filled or not. The return value can also be a list of
            paths and list of booleans of the same length.
            """
        raise NotImplementedError('Derived must override')

    def __call__(self, path, mutation_size, linewidth, aspect_ratio=1.0):
        """
            The __call__ method is a thin wrapper around the transmute method
            and takes care of the aspect ratio.
            """
        if aspect_ratio is not None:
            vertices = path.vertices / [1, aspect_ratio]
            path_shrunk = Path(vertices, path.codes)
            path_mutated, fillable = self.transmute(path_shrunk, mutation_size, linewidth)
            if np.iterable(fillable):
                path_list = [Path(p.vertices * [1, aspect_ratio], p.codes) for p in path_mutated]
                return (path_list, fillable)
            else:
                return (path_mutated, fillable)
        else:
            return self.transmute(path, mutation_size, linewidth)
