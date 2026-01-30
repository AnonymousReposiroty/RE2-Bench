from sklearn import get_config

def _routing_enabled():
    """Return whether metadata routing is enabled.

    .. versionadded:: 1.3

    Returns
    -------
    enabled : bool
        Whether metadata routing is enabled. If the config is not set, it
        defaults to False.
    """
    return get_config().get("enable_metadata_routing", False)
