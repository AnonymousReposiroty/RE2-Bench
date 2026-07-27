from pathlib import Path
import PIL.Image
import PIL.PngImagePlugin
from urllib import parse

def imread(fname, format=None):
    """
    Read an image from a file into an array.

    .. note::

        This function exists for historical reasons.  It is recommended to
        use `PIL.Image.open` instead for loading images.

    Parameters
    ----------
    fname : str or file-like
        The image file to read: a filename, a URL or a file-like object opened
        in read-binary mode.

        Passing a URL is deprecated.  Please open the URL
        for reading and pass the result to Pillow, e.g. with
        ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.
    format : str, optional
        The image file format assumed for reading the data.  The image is
        loaded as a PNG file if *format* is set to "png", if *fname* is a path
        or opened file with a ".png" extension, or if it is a URL.  In all
        other cases, *format* is ignored and the format is auto-detected by
        `PIL.Image.open`.

    Returns
    -------
    `numpy.array`
        The image data. The returned array has shape

        - (M, N) for grayscale images.
        - (M, N, 3) for RGB images.
        - (M, N, 4) for RGBA images.

        PNG images are returned as float arrays (0-1).  All other formats are
        returned as int arrays, with a bit depth determined by the file's
        contents.
    """
    # hide imports to speed initial import on systems with slow linkers
    from urllib import parse

    if format is None:
        if isinstance(fname, str):
            parsed = parse.urlparse(fname)
            # If the string is a URL (Windows paths appear as if they have a
            # length-1 scheme), assume png.
            if len(parsed.scheme) > 1:
                ext = 'png'
            else:
                ext = Path(fname).suffix.lower()[1:]
        elif hasattr(fname, 'geturl'):  # Returned by urlopen().
            # We could try to parse the url's path and use the extension, but
            # returning png is consistent with the block above.  Note that this
            # if clause has to come before checking for fname.name as
            # urlopen("file:///...") also has a name attribute (with the fixed
            # value "<urllib response>").
            ext = 'png'
        elif hasattr(fname, 'name'):
            ext = Path(fname.name).suffix.lower()[1:]
        else:
            ext = 'png'
    else:
        ext = format
    img_open = (
        PIL.PngImagePlugin.PngImageFile if ext == 'png' else PIL.Image.open)
    if isinstance(fname, str) and len(parse.urlparse(fname).scheme) > 1:
        # Pillow doesn't handle URLs directly.
        raise ValueError(
            "Please open the URL for reading and pass the "
            "result to Pillow, e.g. with "
            "``np.array(PIL.Image.open(urllib.request.urlopen(url)))``."
            )
    with img_open(fname) as image:
        return (_pil_png_to_float_array(image)
                if isinstance(image, PIL.PngImagePlugin.PngImageFile) else
                pil_to_array(image))
