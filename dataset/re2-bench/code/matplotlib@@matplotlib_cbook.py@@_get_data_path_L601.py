from pathlib import Path
import matplotlib

def _get_data_path(*args):
    """
    Return the `pathlib.Path` to a resource file provided by Matplotlib.

    ``*args`` specify a path relative to the base data path.
    """
    return Path(matplotlib.get_data_path(), *args)
