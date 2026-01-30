class TransformNode:
    """
    The base class for anything that participates in the transform tree
    and needs to invalidate its parents or be invalidated.  This includes
    classes that are not really transforms, such as bounding boxes, since some
    transforms depend on bounding boxes to compute their values.
    """
    _VALID, _INVALID_AFFINE_ONLY, _INVALID_FULL = range(3)
    is_affine = False
    pass_through = False
    "\n    If pass_through is True, all ancestors will always be\n    invalidated, even if 'self' is already invalid.\n    "

    def __init__(self, shorthand_name=None):
        """
        Parameters
        ----------
        shorthand_name : str
            A string representing the "name" of the transform. The name carries
            no significance other than to improve the readability of
            ``str(transform)`` when DEBUG=True.
        """
        self._parents = {}
        self._invalid = self._INVALID_FULL
        self._shorthand_name = shorthand_name or ''
    if DEBUG:

        def __str__(self):
            return self._shorthand_name or repr(self)

    def invalidate(self):
        """
        Invalidate this `TransformNode` and triggers an invalidation of its
        ancestors.  Should be called any time the transform changes.
        """
        return self._invalidate_internal(level=self._INVALID_AFFINE_ONLY if self.is_affine else self._INVALID_FULL, invalidating_node=self)

    def _invalidate_internal(self, level, invalidating_node):
        """
        Called by :meth:`invalidate` and subsequently ascends the transform
        stack calling each TransformNode's _invalidate_internal method.
        """
        if level <= self._invalid and (not self.pass_through):
            return
        self._invalid = level
        for parent in list(self._parents.values()):
            parent = parent()
            if parent is not None:
                parent._invalidate_internal(level=level, invalidating_node=self)
