import functools
import os
from pathlib import Path
import re
import subprocess
import sys
from packaging.version import parse as parse_version
from . import _api, _version, cbook, _docstring, rcsetup
import winreg

@functools.cache
def _get_executable_info(name):
    """
    Get the version of some executable that Matplotlib optionally depends on.

    .. warning::
       The list of executables that this function supports is set according to
       Matplotlib's internal needs, and may change without notice.

    Parameters
    ----------
    name : str
        The executable to query.  The following values are currently supported:
        "dvipng", "gs", "inkscape", "magick", "pdftocairo", "pdftops".  This
        list is subject to change without notice.

    Returns
    -------
    tuple
        A namedtuple with fields ``executable`` (`str`) and ``version``
        (`packaging.Version`, or ``None`` if the version cannot be determined).

    Raises
    ------
    ExecutableNotFoundError
        If the executable is not found or older than the oldest version
        supported by Matplotlib.  For debugging purposes, it is also
        possible to "hide" an executable from Matplotlib by adding it to the
        :envvar:`_MPLHIDEEXECUTABLES` environment variable (a comma-separated
        list), which must be set prior to any calls to this function.
    ValueError
        If the executable is not one that we know how to query.
    """

    def impl(args, regex, min_ver=None, ignore_exit_code=False):
        # Execute the subprocess specified by args; capture stdout and stderr.
        # Search for a regex match in the output; if the match succeeds, the
        # first group of the match is the version.
        # Return an _ExecInfo if the executable exists, and has a version of
        # at least min_ver (if set); else, raise ExecutableNotFoundError.
        try:
            output = subprocess.check_output(
                args, stderr=subprocess.STDOUT,
                text=True, errors="replace", timeout=30)
        except subprocess.CalledProcessError as _cpe:
            if ignore_exit_code:
                output = _cpe.output
            else:
                raise ExecutableNotFoundError(str(_cpe)) from _cpe
        except subprocess.TimeoutExpired as _te:
            msg = f"Timed out running {cbook._pformat_subprocess(args)}"
            raise ExecutableNotFoundError(msg) from _te
        except OSError as _ose:
            raise ExecutableNotFoundError(str(_ose)) from _ose
        match = re.search(regex, output)
        if match:
            raw_version = match.group(1)
            version = parse_version(raw_version)
            if min_ver is not None and version < parse_version(min_ver):
                raise ExecutableNotFoundError(
                    f"You have {args[0]} version {version} but the minimum "
                    f"version supported by Matplotlib is {min_ver}")
            return _ExecInfo(args[0], raw_version, version)
        else:
            raise ExecutableNotFoundError(
                f"Failed to determine the version of {args[0]} from "
                f"{' '.join(args)}, which output {output}")

    if name in os.environ.get("_MPLHIDEEXECUTABLES", "").split(","):
        raise ExecutableNotFoundError(f"{name} was hidden")

    if name == "dvipng":
        return impl(["dvipng", "-version"], "(?m)^dvipng(?: .*)? (.+)", "1.6")
    elif name == "gs":
        execs = (["gswin32c", "gswin64c", "mgs", "gs"]  # "mgs" for miktex.
                 if sys.platform == "win32" else
                 ["gs"])
        for e in execs:
            try:
                return impl([e, "--version"], "(.*)", "9")
            except ExecutableNotFoundError:
                pass
        message = "Failed to find a Ghostscript installation"
        raise ExecutableNotFoundError(message)
    elif name == "inkscape":
        try:
            # Try headless option first (needed for Inkscape version < 1.0):
            return impl(["inkscape", "--without-gui", "-V"],
                        "Inkscape ([^ ]*)")
        except ExecutableNotFoundError:
            pass  # Suppress exception chaining.
        # If --without-gui is not accepted, we may be using Inkscape >= 1.0 so
        # try without it:
        return impl(["inkscape", "-V"], "Inkscape ([^ ]*)")
    elif name == "magick":
        if sys.platform == "win32":
            # Check the registry to avoid confusing ImageMagick's convert with
            # Windows's builtin convert.exe.
            import winreg
            binpath = ""
            for flag in [0, winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY]:
                try:
                    with winreg.OpenKeyEx(
                            winreg.HKEY_LOCAL_MACHINE,
                            r"Software\Imagemagick\Current",
                            0, winreg.KEY_QUERY_VALUE | flag) as hkey:
                        binpath = winreg.QueryValueEx(hkey, "BinPath")[0]
                except OSError:
                    pass
            path = None
            if binpath:
                for name in ["convert.exe", "magick.exe"]:
                    candidate = Path(binpath, name)
                    if candidate.exists():
                        path = str(candidate)
                        break
            if path is None:
                raise ExecutableNotFoundError(
                    "Failed to find an ImageMagick installation")
        else:
            path = "convert"
        # Ignore deprecation warning for "convert" on IM>=7.1.1-33.
        info = impl([path, "--version"], r"(?sm:.*^)Version: ImageMagick (\S*)")
        if info.raw_version == "7.0.10-34":
            # https://github.com/ImageMagick/ImageMagick/issues/2720
            raise ExecutableNotFoundError(
                f"You have ImageMagick {info.version}, which is unsupported")
        return info
    elif name == "pdftocairo":
        return impl(["pdftocairo", "-v"], "pdftocairo version (.*)")
    elif name == "pdftops":
        info = impl(["pdftops", "-v"], "^pdftops version (.*)",
                    ignore_exit_code=True)
        if info and not (
                3 <= info.version.major or
                # poppler version numbers.
                parse_version("0.9") <= info.version < parse_version("1.0")):
            raise ExecutableNotFoundError(
                f"You have pdftops version {info.version} but the minimum "
                f"version supported by Matplotlib is 3.0")
        return info
    else:
        raise ValueError(f"Unknown executable: {name!r}")
