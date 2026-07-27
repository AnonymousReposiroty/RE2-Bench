from functools import cache, lru_cache
import os
from pathlib import Path
import threading
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font

class FontManager:
    """
    On import, the `FontManager` singleton instance creates a list of ttf and
    afm fonts and caches their `FontProperties`.  The `FontManager.findfont`
    method does a nearest neighbor search to find the font that most closely
    matches the specification.  If no good enough match is found, the default
    font is returned.

    Fonts added with the `FontManager.addfont` method will not persist in the
    cache; therefore, `addfont` will need to be called every time Matplotlib is
    imported. This method should only be used if and when a font cannot be
    installed on your operating system by other means.

    Notes
    -----
    The `FontManager.addfont` method must be called on the global `FontManager`
    instance.

    Example usage::

        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        font_dirs = ["/resources/fonts"]  # The path to the custom font file.
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
    """
    __version__ = '3.11.0a1'

    def __init__(self, size=None, weight='normal'):
        self._version = self.__version__
        self.__default_weight = weight
        self.default_size = size
        paths = [cbook._get_data_path('fonts', subdir) for subdir in ['ttf', 'afm', 'pdfcorefonts']]
        _log.debug('font search path %s', paths)
        self.defaultFamily = {'ttf': 'DejaVu Sans', 'afm': 'Helvetica'}
        self.afmlist = []
        self.ttflist = []
        timer = threading.Timer(5, lambda: _log.warning('Matplotlib is building the font cache; this may take a moment.'))
        timer.start()
        try:
            for fontext in ['afm', 'ttf']:
                for path in [*findSystemFonts(paths, fontext=fontext), *findSystemFonts(fontext=fontext)]:
                    try:
                        self.addfont(path)
                    except OSError as exc:
                        _log.info('Failed to open font file %s: %s', path, exc)
                    except Exception as exc:
                        _log.info('Failed to extract font properties from %s: %s', path, exc)
        finally:
            timer.cancel()

    def addfont(self, path):
        """
        Cache the properties of the font at *path* to make it available to the
        `FontManager`.  The type of font is inferred from the path suffix.

        Parameters
        ----------
        path : str or path-like

        Notes
        -----
        This method is useful for adding a custom font without installing it in
        your operating system. See the `FontManager` singleton instance for
        usage and caveats about this function.
        """
        path = os.fsdecode(path)
        if Path(path).suffix.lower() == '.afm':
            with open(path, 'rb') as fh:
                font = _afm.AFM(fh)
            prop = afmFontProperty(path, font)
            self.afmlist.append(prop)
        else:
            font = ft2font.FT2Font(path)
            prop = ttfFontProperty(font)
            self.ttflist.append(prop)
        self._findfont_cached.cache_clear()

    @staticmethod
    def _expand_aliases(family):
        if family in ('sans', 'sans serif'):
            family = 'sans-serif'
        return mpl.rcParams['font.' + family]

    def score_family(self, families, family2):
        """
        Return a match score between the list of font families in
        *families* and the font family name *family2*.

        An exact match at the head of the list returns 0.0.

        A match further down the list will return between 0 and 1.

        No match will return 1.0.
        """
        if not isinstance(families, (list, tuple)):
            families = [families]
        elif len(families) == 0:
            return 1.0
        family2 = family2.lower()
        step = 1 / len(families)
        for i, family1 in enumerate(families):
            family1 = family1.lower()
            if family1 in font_family_aliases:
                options = [*map(str.lower, self._expand_aliases(family1))]
                if family2 in options:
                    idx = options.index(family2)
                    return (i + idx / len(options)) * step
            elif family1 == family2:
                return i * step
        return 1.0

    def score_style(self, style1, style2):
        """
        Return a match score between *style1* and *style2*.

        An exact match returns 0.0.

        A match between 'italic' and 'oblique' returns 0.1.

        No match returns 1.0.
        """
        if style1 == style2:
            return 0.0
        elif style1 in ('italic', 'oblique') and style2 in ('italic', 'oblique'):
            return 0.1
        return 1.0

    def score_variant(self, variant1, variant2):
        """
        Return a match score between *variant1* and *variant2*.

        An exact match returns 0.0, otherwise 1.0.
        """
        if variant1 == variant2:
            return 0.0
        else:
            return 1.0

    def score_stretch(self, stretch1, stretch2):
        """
        Return a match score between *stretch1* and *stretch2*.

        The result is the absolute value of the difference between the
        CSS numeric values of *stretch1* and *stretch2*, normalized
        between 0.0 and 1.0.
        """
        try:
            stretchval1 = int(stretch1)
        except ValueError:
            stretchval1 = stretch_dict.get(stretch1, 500)
        try:
            stretchval2 = int(stretch2)
        except ValueError:
            stretchval2 = stretch_dict.get(stretch2, 500)
        return abs(stretchval1 - stretchval2) / 1000.0

    def score_weight(self, weight1, weight2):
        """
        Return a match score between *weight1* and *weight2*.

        The result is 0.0 if both weight1 and weight 2 are given as strings
        and have the same value.

        Otherwise, the result is the absolute value of the difference between
        the CSS numeric values of *weight1* and *weight2*, normalized between
        0.05 and 1.0.
        """
        if cbook._str_equal(weight1, weight2):
            return 0.0
        w1 = _normalize_weight(weight1)
        w2 = _normalize_weight(weight2)
        return 0.95 * (abs(w1 - w2) / 1000) + 0.05

    def score_size(self, size1, size2):
        """
        Return a match score between *size1* and *size2*.

        If *size2* (the size specified in the font file) is 'scalable', this
        function always returns 0.0, since any font size can be generated.

        Otherwise, the result is the absolute distance between *size1* and
        *size2*, normalized so that the usual range of font sizes (6pt -
        72pt) will lie between 0.0 and 1.0.
        """
        if size2 == 'scalable':
            return 0.0
        try:
            sizeval1 = float(size1)
        except ValueError:
            sizeval1 = self.default_size * font_scalings[size1]
        try:
            sizeval2 = float(size2)
        except ValueError:
            return 1.0
        return abs(sizeval1 - sizeval2) / 72

    def findfont(self, prop, fontext='ttf', directory=None, fallback_to_default=True, rebuild_if_missing=True):
        """
        Find the path to the font file most closely matching the given font properties.

        Parameters
        ----------
        prop : str or `~matplotlib.font_manager.FontProperties`
            The font properties to search for. This can be either a
            `.FontProperties` object or a string defining a
            `fontconfig patterns`_.

        fontext : {'ttf', 'afm'}, default: 'ttf'
            The extension of the font file:

            - 'ttf': TrueType and OpenType fonts (.ttf, .ttc, .otf)
            - 'afm': Adobe Font Metrics (.afm)

        directory : str, optional
            If given, only search this directory and its subdirectories.

        fallback_to_default : bool
            If True, will fall back to the default font family (usually
            "DejaVu Sans" or "Helvetica") if the first lookup hard-fails.

        rebuild_if_missing : bool
            Whether to rebuild the font cache and search again if the first
            match appears to point to a nonexisting font (i.e., the font cache
            contains outdated entries).

        Returns
        -------
        str
            The filename of the best matching font.

        Notes
        -----
        This performs a nearest neighbor search.  Each font is given a
        similarity score to the target font properties.  The first font with
        the highest score is returned.  If no matches below a certain
        threshold are found, the default font (usually DejaVu Sans) is
        returned.

        The result is cached, so subsequent lookups don't have to
        perform the O(n) nearest neighbor search.

        See the `W3C Cascading Style Sheet, Level 1
        <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation
        for a description of the font finding algorithm.

        .. _fontconfig patterns:
           https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
        """
        rc_params = tuple((tuple(mpl.rcParams[key]) for key in ['font.serif', 'font.sans-serif', 'font.cursive', 'font.fantasy', 'font.monospace']))
        ret = self._findfont_cached(prop, fontext, directory, fallback_to_default, rebuild_if_missing, rc_params)
        if isinstance(ret, cbook._ExceptionInfo):
            raise ret.to_exception()
        return ret

    @lru_cache(1024)
    def _findfont_cached(self, prop, fontext, directory, fallback_to_default, rebuild_if_missing, rc_params):
        prop = FontProperties._from_any(prop)
        fname = prop.get_file()
        if fname is not None:
            return fname
        if fontext == 'afm':
            fontlist = self.afmlist
        else:
            fontlist = self.ttflist
        best_score = 1e+64
        best_font = None
        _log.debug('findfont: Matching %s.', prop)
        for font in fontlist:
            if directory is not None and Path(directory) not in Path(font.fname).parents:
                continue
            score = self.score_family(prop.get_family(), font.name) * 10 + self.score_style(prop.get_style(), font.style) + self.score_variant(prop.get_variant(), font.variant) + self.score_weight(prop.get_weight(), font.weight) + self.score_stretch(prop.get_stretch(), font.stretch) + self.score_size(prop.get_size(), font.size)
            _log.debug('findfont: score(%s) = %s', font, score)
            if score < best_score:
                best_score = score
                best_font = font
            if score == 0:
                break
        if best_font is not None and _normalize_weight(prop.get_weight()) != _normalize_weight(best_font.weight):
            _log.warning('findfont: Failed to find font weight %s, now using %s.', prop.get_weight(), best_font.weight)
        if best_font is None or best_score >= 10.0:
            if fallback_to_default:
                _log.warning('findfont: Font family %s not found. Falling back to %s.', prop.get_family(), self.defaultFamily[fontext])
                for family in map(str.lower, prop.get_family()):
                    if family in font_family_aliases:
                        _log.warning('findfont: Generic family %r not found because none of the following families were found: %s', family, ', '.join(self._expand_aliases(family)))
                default_prop = prop.copy()
                default_prop.set_family(self.defaultFamily[fontext])
                return self.findfont(default_prop, fontext, directory, fallback_to_default=False)
            else:
                return cbook._ExceptionInfo(ValueError, f'Failed to find font {prop}, and fallback to the default font was disabled')
        else:
            _log.debug('findfont: Matching %s to %s (%r) with score of %f.', prop, best_font.name, best_font.fname, best_score)
            result = best_font.fname
        if not os.path.isfile(result):
            if rebuild_if_missing:
                _log.info('findfont: Found a missing font file.  Rebuilding cache.')
                new_fm = _load_fontmanager(try_read_cache=False)
                vars(self).update(vars(new_fm))
                return self.findfont(prop, fontext, directory, rebuild_if_missing=False)
            else:
                return cbook._ExceptionInfo(ValueError, 'No valid font could be found')
        return _cached_realpath(result)
