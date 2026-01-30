def _max_precision_float_dtype(xp, device):
    """Return the float dtype with the highest precision supported by the device."""
    # TODO: Update to use `__array_namespace__info__()` from array-api v2023.12
    # when/if that becomes more widespread.
    if _is_xp_namespace(xp, "torch") and str(device).startswith(
        "mps"
    ):  # pragma: no cover
        return xp.float32
    return xp.float64
