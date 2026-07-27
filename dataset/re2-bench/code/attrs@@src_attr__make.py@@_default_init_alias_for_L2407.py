def _default_init_alias_for(name: str) -> str:
    """
    The default __init__ parameter name for a field.

    This performs private-name adjustment via leading-unscore stripping,
    and is the default value of Attribute.alias if not provided.
    """

    return name.lstrip("_")
