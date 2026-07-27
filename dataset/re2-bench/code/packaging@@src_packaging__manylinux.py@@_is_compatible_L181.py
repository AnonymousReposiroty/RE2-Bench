import _manylinux  # noqa: PLC0415

def _is_compatible(arch: str, version: _GLibCVersion) -> bool:
    sys_glibc = _get_glibc_version()
    if sys_glibc < version:
        return False
    # Check for presence of _manylinux module.
    try:
        import _manylinux  # noqa: PLC0415
    except ImportError:
        return True
    if hasattr(_manylinux, "manylinux_compatible"):
        result = _manylinux.manylinux_compatible(version[0], version[1], arch)
        if result is not None:
            return bool(result)
        return True
    if version == _GLibCVersion(2, 5) and hasattr(_manylinux, "manylinux1_compatible"):
        return bool(_manylinux.manylinux1_compatible)
    if version == _GLibCVersion(2, 12) and hasattr(
        _manylinux, "manylinux2010_compatible"
    ):
        return bool(_manylinux.manylinux2010_compatible)
    if version == _GLibCVersion(2, 17) and hasattr(
        _manylinux, "manylinux2014_compatible"
    ):
        return bool(_manylinux.manylinux2014_compatible)
    return True
