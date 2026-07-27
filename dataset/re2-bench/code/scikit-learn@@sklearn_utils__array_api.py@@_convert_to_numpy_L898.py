import numpy

def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray on the CPU."""
    if _is_xp_namespace(xp, "torch"):
        return array.cpu().numpy()
    elif _is_xp_namespace(xp, "cupy"):  # pragma: nocover
        return array.get()
    elif _is_xp_namespace(xp, "array_api_strict"):
        return numpy.asarray(xp.asarray(array, device=xp.Device("CPU_DEVICE")))

    return numpy.asarray(array)
