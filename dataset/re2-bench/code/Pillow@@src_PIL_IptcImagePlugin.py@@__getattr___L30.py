from ._deprecate import deprecate

def __getattr__(name: str) -> bytes:
    if name == "PAD":
        deprecate("IptcImagePlugin.PAD", 12)
        return b"\0\0\0\0"
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
