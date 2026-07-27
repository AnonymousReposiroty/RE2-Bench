import numpy as np
from matplotlib import (
    _api, backend_tools as tools, cbook, colors, _docstring, text,
    _tight_bbox, transforms, widgets, is_interactive, rcParams)
from matplotlib._enums import JoinStyle, CapStyle

class GraphicsContextBase:
    """An abstract base class that provides color, line styles, etc."""

    def __init__(self):
        self._alpha = 1.0
        self._forced_alpha = False
        self._antialiased = 1
        self._capstyle = CapStyle('butt')
        self._cliprect = None
        self._clippath = None
        self._dashes = (0, None)
        self._joinstyle = JoinStyle('round')
        self._linestyle = 'solid'
        self._linewidth = 1
        self._rgb = (0.0, 0.0, 0.0, 1.0)
        self._hatch = None
        self._hatch_color = None
        self._hatch_linewidth = rcParams['hatch.linewidth']
        self._url = None
        self._gid = None
        self._snap = None
        self._sketch = None

    def get_clip_path(self):
        """
        Return the clip path in the form (path, transform), where path
        is a `~.path.Path` instance, and transform is
        an affine transform to apply to the path before clipping.
        """
        if self._clippath is not None:
            tpath, tr = self._clippath.get_transformed_path_and_affine()
            if np.all(np.isfinite(tpath.vertices)):
                return (tpath, tr)
            else:
                _log.warning('Ill-defined clip_path detected. Returning None.')
                return (None, None)
        return (None, None)
