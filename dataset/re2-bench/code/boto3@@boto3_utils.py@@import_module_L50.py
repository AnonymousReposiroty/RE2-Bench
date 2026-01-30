import sys

def import_module(name):
    """Import module given a name.

    Does not support relative imports.

    """
    __import__(name)
    return sys.modules[name]
