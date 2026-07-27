import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale, _image

class MultiNorm(Norm):
    """
    A class which contains multiple scalar norms.
    """

    def __init__(self, norms, vmin=None, vmax=None, clip=None):
        """
        Parameters
        ----------
        norms : list of (str or `Normalize`)
            The constituent norms. The list must have a minimum length of 1.
        vmin, vmax : None or list of (float or None)
            Limits of the constituent norms.
            If a list, one value is assigned to each of the constituent
            norms.
            If None, the limits of the constituent norms
            are not changed.
        clip : None or list of bools, default: None
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]`` for the constituent norms.
            If a list, each value is assigned to each of the constituent
            norms.
            If None, the behaviour of the constituent norms is not changed.
        """
        if cbook.is_scalar_or_string(norms):
            raise ValueError(f'MultiNorm must be assigned an iterable of norms, where each norm is of type `str`, or `Normalize`, not {type(norms)}')
        if len(norms) < 1:
            raise ValueError('MultiNorm must be assigned at least one norm')

        def resolve(norm):
            if isinstance(norm, str):
                scale_cls = _api.check_getitem(scale._scale_mapping, norm=norm)
                return mpl.colorizer._auto_norm_from_scale(scale_cls)()
            elif isinstance(norm, Normalize):
                return norm
            else:
                raise ValueError(f'Each norm assigned to MultiNorm must be of type `str`, or `Normalize`, not {type(norm)}')
        self._norms = tuple((resolve(norm) for norm in norms))
        self.callbacks = cbook.CallbackRegistry(signals=['changed'])
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
        for n in self._norms:
            n.callbacks.connect('changed', self._changed)

    @property
    def n_components(self):
        """Number of norms held by this `MultiNorm`."""
        return len(self._norms)

    @property
    def norms(self):
        """The individual norms held by this `MultiNorm`."""
        return self._norms

    @property
    def vmin(self):
        """The lower limit of each constituent norm."""
        return tuple((n.vmin for n in self._norms))

    @vmin.setter
    def vmin(self, values):
        if values is None:
            return
        if not np.iterable(values) or len(values) != self.n_components:
            raise ValueError(f'*vmin* must have one component for each norm. Expected an iterable of length {self.n_components}, but got {values!r}')
        with self.callbacks.blocked():
            for norm, v in zip(self.norms, values):
                norm.vmin = v
        self._changed()

    @property
    def vmax(self):
        """The upper limit of each constituent norm."""
        return tuple((n.vmax for n in self._norms))

    @vmax.setter
    def vmax(self, values):
        if values is None:
            return
        if not np.iterable(values) or len(values) != self.n_components:
            raise ValueError(f'*vmax* must have one component for each norm. Expected an iterable of length {self.n_components}, but got {values!r}')
        with self.callbacks.blocked():
            for norm, v in zip(self.norms, values):
                norm.vmax = v
        self._changed()

    @property
    def clip(self):
        """The clip behaviour of each constituent norm."""
        return tuple((n.clip for n in self._norms))

    @clip.setter
    def clip(self, values):
        if values is None:
            return
        if not np.iterable(values) or len(values) != self.n_components:
            raise ValueError(f'*clip* must have one component for each norm. Expected an iterable of length {self.n_components}, but got {values!r}')
        with self.callbacks.blocked():
            for norm, v in zip(self.norms, values):
                norm.clip = v
        self._changed()

    def _changed(self):
        """
        Call this whenever the norm is changed to notify all the
        callback listeners to the 'changed' signal.
        """
        self.callbacks.process('changed')
