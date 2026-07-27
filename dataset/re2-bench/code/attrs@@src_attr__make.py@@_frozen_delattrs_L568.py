from .exceptions import (
    DefaultAlreadySetError,
    FrozenInstanceError,
    NotAnAttrsClassError,
    UnannotatedAttributeError,
)

def _frozen_delattrs(self, name):
    """
    Attached to frozen classes as __delattr__.
    """
    if isinstance(self, BaseException) and name in ("__notes__",):
        BaseException.__delattr__(self, name)
        return

    raise FrozenInstanceError
