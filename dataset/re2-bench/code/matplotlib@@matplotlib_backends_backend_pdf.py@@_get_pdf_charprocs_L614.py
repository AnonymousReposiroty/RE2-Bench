import numpy as np
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib.ft2font import FT2Font, FaceFlags, Kerning, LoadFlags, StyleFlags
from matplotlib.path import Path
from matplotlib import _path

def _get_pdf_charprocs(font_path, glyph_ids):
    font = get_font(font_path, hinting_factor=1)
    conv = 1000 / font.units_per_EM  # Conversion to PS units (1/1000's).
    procs = {}
    for glyph_id in glyph_ids:
        g = font.load_glyph(glyph_id, LoadFlags.NO_SCALE)
        # NOTE: We should be using round(), but instead use
        # "(x+.5).astype(int)" to keep backcompat with the old ttconv code
        # (this is different for negative x's).
        d1 = (np.array([g.horiAdvance, 0, *g.bbox]) * conv + .5).astype(int)
        v, c = font.get_path()
        v = (v * 64).astype(int)  # Back to TrueType's internal units (1/64's).
        # Backcompat with old ttconv code: control points between two quads are
        # omitted if they are exactly at the midpoint between the control of
        # the quad before and the quad after, but ttconv used to interpolate
        # *after* conversion to PS units, causing floating point errors.  Here
        # we reproduce ttconv's logic, detecting these "implicit" points and
        # re-interpolating them.  Note that occasionally (e.g. with DejaVu Sans
        # glyph "0") a point detected as "implicit" is actually explicit, and
        # will thus be shifted by 1.
        quads, = np.nonzero(c == 3)
        quads_on = quads[1::2]
        quads_mid_on = np.array(
            sorted({*quads_on} & {*(quads - 1)} & {*(quads + 1)}), int)
        implicit = quads_mid_on[
            (v[quads_mid_on]  # As above, use astype(int), not // division
             == ((v[quads_mid_on - 1] + v[quads_mid_on + 1]) / 2).astype(int))
            .all(axis=1)]
        if (font.postscript_name, glyph_id) in [
                ("DejaVuSerif-Italic", 77),  # j
                ("DejaVuSerif-Italic", 135),  # \AA
        ]:
            v[:, 0] -= 1  # Hard-coded backcompat (FreeType shifts glyph by 1).
        v = (v * conv + .5).astype(int)  # As above re: truncation vs rounding.
        v[implicit] = ((  # Fix implicit points; again, truncate.
            (v[implicit - 1] + v[implicit + 1]) / 2).astype(int))
        procs[font.get_glyph_name(glyph_id)] = (
            " ".join(map(str, d1)).encode("ascii") + b" d1\n"
            + _path.convert_to_string(
                Path(v, c), None, None, False, None, -1,
                # no code for quad Beziers triggers auto-conversion to cubics.
                [b"m", b"l", b"", b"c", b"h"], True)
            + b"f")
    return procs
