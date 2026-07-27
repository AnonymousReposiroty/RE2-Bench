import os
import shutil
from tempfile import TemporaryDirectory
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib.texmanager import TexManager

def _convert_psfrags(tmppath, psfrags, paper_width, paper_height, orientation):
    """
    When we want to use the LaTeX backend with postscript, we write PSFrag tags
    to a temporary postscript file, each one marking a position for LaTeX to
    render some text. convert_psfrags generates a LaTeX document containing the
    commands to convert those tags to text. LaTeX/dvips produces the postscript
    file that includes the actual text.
    """
    with mpl.rc_context({
            "text.latex.preamble":
            mpl.rcParams["text.latex.preamble"] +
            mpl.texmanager._usepackage_if_not_loaded("color") +
            mpl.texmanager._usepackage_if_not_loaded("graphicx") +
            mpl.texmanager._usepackage_if_not_loaded("psfrag") +
            r"\geometry{papersize={%(width)sin,%(height)sin},margin=0in}"
            % {"width": paper_width, "height": paper_height}
    }):
        dvifile = TexManager().make_dvi(
            "\n"
            r"\begin{figure}""\n"
            r"  \centering\leavevmode""\n"
            r"  %(psfrags)s""\n"
            r"  \includegraphics*[angle=%(angle)s]{%(epsfile)s}""\n"
            r"\end{figure}"
            % {
                "psfrags": "\n".join(psfrags),
                "angle": 90 if orientation == 'landscape' else 0,
                "epsfile": tmppath.resolve().as_posix(),
            },
            fontsize=10)  # tex's default fontsize.

    with TemporaryDirectory() as tmpdir:
        psfile = os.path.join(tmpdir, "tmp.ps")
        cbook._check_and_log_subprocess(
            ['dvips', '-q', '-R0', '-o', psfile, dvifile], _log)
        shutil.move(psfile, tmppath)

    # check if the dvips created a ps in landscape paper.  Somehow,
    # above latex+dvips results in a ps file in a landscape mode for a
    # certain figure sizes (e.g., 8.3in, 5.8in which is a5). And the
    # bounding box of the final output got messed up. We check see if
    # the generated ps file is in landscape and return this
    # information. The return value is used in pstoeps step to recover
    # the correct bounding box. 2010-06-05 JJL
    with open(tmppath) as fh:
        psfrag_rotated = "Landscape" in fh.read(1000)
    return psfrag_rotated
