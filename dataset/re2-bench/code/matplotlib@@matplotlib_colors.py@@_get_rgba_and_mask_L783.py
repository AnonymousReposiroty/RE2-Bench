import numpy as np

class Colormap:
    """
    Baseclass for all scalar to RGBA mappings.

    Typically, Colormap instances are used to convert data values (floats)
    from the interval ``[0, 1]`` to the RGBA color that the respective
    Colormap represents. For scaling of data into the ``[0, 1]`` interval see
    `matplotlib.colors.Normalize`. Subclasses of `matplotlib.cm.ScalarMappable`
    make heavy use of this ``data -> normalize -> map-to-color`` processing
    chain.
    """

    def __init__(self, name, N=256, *, bad=None, under=None, over=None):
        """
        Parameters
        ----------
        name : str
            The name of the colormap.
        N : int
            The number of RGB quantization levels.
        bad : :mpltype:`color`, default: transparent
            The color for invalid values (NaN or masked).

            .. versionadded:: 3.11

        under : :mpltype:`color`, default: color of the lowest value
            The color for low out-of-range values.

            .. versionadded:: 3.11

        over : :mpltype:`color`, default: color of the highest value
            The color for high out-of-range values.

            .. versionadded:: 3.11
        """
        self.name = name
        self.N = int(N)
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0) if bad is None else to_rgba(bad)
        self._rgba_under = None if under is None else to_rgba(under)
        self._rgba_over = None if over is None else to_rgba(over)
        self._i_under = self.N
        self._i_over = self.N + 1
        self._i_bad = self.N + 2
        self._isinit = False
        self.n_variates = 1
        self.colorbar_extend = False

    def _get_rgba_and_mask(self, X, alpha=None, bytes=False):
        """
        Parameters
        ----------
        X : float or int or array-like
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool, default: False
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\\s in the
            interval ``[0, 255]``.

        Returns
        -------
        colors : np.ndarray
            Array of RGBA values with a shape of ``X.shape + (4, )``.
        mask : np.ndarray
            Boolean array with True where the input is ``np.nan`` or masked.
        """
        self._ensure_inited()
        xa = np.array(X, copy=True)
        if not xa.dtype.isnative:
            xa = xa.byteswap().view(xa.dtype.newbyteorder())
        if xa.dtype.kind == 'f':
            xa *= self.N
            xa[xa == self.N] = self.N - 1
        mask_under = xa < 0
        mask_over = xa >= self.N
        mask_bad = X.mask if np.ma.is_masked(X) else np.isnan(xa)
        with np.errstate(invalid='ignore'):
            xa = xa.astype(int)
        xa[mask_under] = self._i_under
        xa[mask_over] = self._i_over
        xa[mask_bad] = self._i_bad
        lut = self._lut
        if bytes:
            lut = (lut * 255).astype(np.uint8)
        rgba = lut.take(xa, axis=0, mode='clip')
        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
            if bytes:
                alpha *= 255
            if alpha.shape not in [(), xa.shape]:
                raise ValueError(f'alpha is array-like but its shape {alpha.shape} does not match that of X {xa.shape}')
            rgba[..., -1] = alpha
            if (lut[-1] == 0).all():
                rgba[mask_bad] = (0, 0, 0, 0)
        return (rgba, mask_bad)

    def _init(self):
        """Generate the lookup table, ``self._lut``."""
        raise NotImplementedError('Abstract class only')

    def _ensure_inited(self):
        if not self._isinit:
            self._init()
