import re
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

def _process_plot_format(fmt, *, ambiguous_fmt_datakey=False):
    """
    Convert a MATLAB style color/line style format string to a (*linestyle*,
    *marker*, *color*) tuple.

    Example format strings include:

    * 'ko': black circles
    * '.b': blue dots
    * 'r--': red dashed lines
    * 'C2--': the third color in the color cycle, dashed lines

    The format is absolute in the sense that if a linestyle or marker is not
    defined in *fmt*, there is no line or marker. This is expressed by
    returning 'None' for the respective quantity.

    See Also
    --------
    matplotlib.Line2D.lineStyles, matplotlib.colors.cnames
        All possible styles and color format strings.
    """

    linestyle = None
    marker = None
    color = None

    # First check whether fmt is just a colorspec, but specifically exclude the
    # grayscale string "1" (not "1.0"), which is interpreted as the tri_down
    # marker "1".  The grayscale string "0" could be unambiguously understood
    # as a color (black) but also excluded for consistency.
    if fmt not in ["0", "1"]:
        try:
            color = mcolors.to_rgba(fmt)
            return linestyle, marker, color
        except ValueError:
            pass

    errfmt = ("{!r} is neither a data key nor a valid format string ({})"
              if ambiguous_fmt_datakey else
              "{!r} is not a valid format string ({})")

    i = 0
    while i < len(fmt):
        c = fmt[i]
        if fmt[i:i+2] in mlines.lineStyles:  # First, the two-char styles.
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            linestyle = fmt[i:i+2]
            i += 2
        elif c in mlines.lineStyles:
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            linestyle = c
            i += 1
        elif c in mlines.lineMarkers:
            if marker is not None:
                raise ValueError(errfmt.format(fmt, "two marker symbols"))
            marker = c
            i += 1
        elif c in mcolors.get_named_colors_mapping():
            if color is not None:
                raise ValueError(errfmt.format(fmt, "two color symbols"))
            color = c
            i += 1
        elif c == "C":
            cn_color = re.match(r"C\d+", fmt[i:])
            if not cn_color:
                raise ValueError(errfmt.format(fmt, "'C' must be followed by a number"))
            color = mcolors.to_rgba(cn_color[0])
            i += len(cn_color[0])
        else:
            raise ValueError(errfmt.format(fmt, f"unrecognized character {c!r}"))

    if linestyle is None and marker is None:
        linestyle = mpl.rcParams['lines.linestyle']
    if linestyle is None:
        linestyle = 'None'
    if marker is None:
        marker = 'None'

    return linestyle, marker, color
