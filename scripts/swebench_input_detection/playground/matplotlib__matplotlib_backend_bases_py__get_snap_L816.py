# Problem: matplotlib@@matplotlib_backend_bases.py@@get_snap_L816
# Module: matplotlib.backend.bases
# Function: get_snap
# Line: 816

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

    def get_snap(self):
        """
        Return the snap setting, which can be:

        * True: snap vertices to the nearest pixel center
        * False: leave vertices as-is
        * None: (auto) If the path contains only rectilinear line segments,
          round to the nearest pixel center
        """
        return self._snap




def test_input(pred_input):
    obj_ins = GraphicsContextBase()
    obj_ins._alpha = 0.25
    obj_ins._forced_alpha = True
    obj_ins._antialiased = 0
    obj_ins._capstyle = 'butt'
    obj_ins._cliprect = None
    obj_ins._clippath = None
    obj_ins._dashes = [0, None]
    obj_ins._joinstyle = 'miter'
    obj_ins._linestyle = 'solid'
    obj_ins._linewidth = 0.0
    obj_ins._rgb = [0.0, 0.0, 0.0, 0.25]
    obj_ins._hatch = None
    obj_ins._hatch_color = None
    obj_ins._hatch_linewidth = 1.0
    obj_ins._url = None
    obj_ins._gid = None
    obj_ins._snap = None
    obj_ins._sketch = None
    obj_ins_pred = GraphicsContextBase()
    obj_ins_pred._alpha = pred_input['self']['_alpha']
    obj_ins_pred._forced_alpha = pred_input['self']['_forced_alpha']
    obj_ins_pred._antialiased = pred_input['self']['_antialiased']
    obj_ins_pred._capstyle = pred_input['self']['_capstyle']
    obj_ins_pred._cliprect = pred_input['self']['_cliprect']
    obj_ins_pred._clippath = pred_input['self']['_clippath']
    obj_ins_pred._dashes = pred_input['self']['_dashes']
    obj_ins_pred._joinstyle = pred_input['self']['_joinstyle']
    obj_ins_pred._linestyle = pred_input['self']['_linestyle']
    obj_ins_pred._linewidth = pred_input['self']['_linewidth']
    obj_ins_pred._rgb = pred_input['self']['_rgb']
    obj_ins_pred._hatch = pred_input['self']['_hatch']
    obj_ins_pred._hatch_color = pred_input['self']['_hatch_color']
    obj_ins_pred._hatch_linewidth = pred_input['self']['_hatch_linewidth']
    obj_ins_pred._url = pred_input['self']['_url']
    obj_ins_pred._gid = pred_input['self']['_gid']
    obj_ins_pred._snap = pred_input['self']['_snap']
    obj_ins_pred._sketch = pred_input['self']['_sketch']
    assert obj_ins.get_snap()==obj_ins_pred.get_snap(), 'Prediction failed!'
