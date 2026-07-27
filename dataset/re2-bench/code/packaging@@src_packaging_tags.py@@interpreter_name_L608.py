import sys

def interpreter_name() -> str:
    """
    Returns the name of the running interpreter.

    Some implementations have a reserved, two-letter abbreviation which will
    be returned when appropriate.
    """
    name = sys.implementation.name
    return INTERPRETER_SHORT_NAMES.get(name) or name
