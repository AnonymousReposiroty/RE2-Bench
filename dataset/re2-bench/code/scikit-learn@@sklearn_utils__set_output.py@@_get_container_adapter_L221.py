def _get_container_adapter(method, estimator=None):
    """Get container adapter."""
    dense_config = _get_output_config(method, estimator)["dense"]
    try:
        return ADAPTERS_MANAGER.adapters[dense_config]
    except KeyError:
        return None
