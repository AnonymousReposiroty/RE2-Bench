import functools

def _register_tool_class(canvas_cls, tool_cls=None):
    """Decorator registering *tool_cls* as a tool class for *canvas_cls*."""
    if tool_cls is None:
        return functools.partial(_register_tool_class, canvas_cls)
    _tool_registry.add((canvas_cls, tool_cls))
    return tool_cls
