import sys

def interpreter_version(*, warn: bool = False) -> str:
    """
    Returns the version of the running interpreter.
    """
    version = _get_config_var("py_version_nodot", warn=warn)
    return str(version) if version else _version_nodot(sys.version_info[:2])
