from functools import cache, reduce, wraps
import inspect
import re
import numpy as np

class ArtistInspector:
    """
    A helper class to inspect an `~matplotlib.artist.Artist` and return
    information about its settable properties and their current values.
    """

    def __init__(self, o):
        """
        Initialize the artist inspector with an `Artist` or an iterable of
        `Artist`\\s.  If an iterable is used, we assume it is a homogeneous
        sequence (all `Artist`\\s are of the same type) and it is your
        responsibility to make sure this is so.
        """
        if not isinstance(o, Artist):
            if np.iterable(o):
                o = list(o)
                if len(o):
                    o = o[0]
        self.oorig = o
        if not isinstance(o, type):
            o = type(o)
        self.o = o
        self.aliasd = self.get_aliases()

    def get_aliases(self):
        """
        Get a dict mapping property fullnames to sets of aliases for each alias
        in the :class:`~matplotlib.artist.ArtistInspector`.

        e.g., for lines::

          {'markerfacecolor': {'mfc'},
           'linewidth'      : {'lw'},
          }
        """
        names = [name for name in dir(self.o) if name.startswith(('set_', 'get_')) and callable(getattr(self.o, name))]
        aliases = {}
        for name in names:
            func = getattr(self.o, name)
            if not self.is_alias(func):
                continue
            propname = re.search(f'`({name[:4]}.*)`', inspect.getdoc(func)).group(1)
            aliases.setdefault(propname[4:], set()).add(name[4:])
        return aliases
    _get_valid_values_regex = re.compile('\\n\\s*(?:\\.\\.\\s+)?ACCEPTS:\\s*((?:.|\\n)*?)(?:$|(?:\\n\\n))')

    def get_valid_values(self, attr):
        """
        Get the legal arguments for the setter associated with *attr*.

        This is done by querying the docstring of the setter for a line that
        begins with "ACCEPTS:" or ".. ACCEPTS:", and then by looking for a
        numpydoc-style documentation for the setter's first argument.
        """
        name = 'set_%s' % attr
        if not hasattr(self.o, name):
            raise AttributeError(f'{self.o} has no function {name}')
        func = getattr(self.o, name)
        if hasattr(func, '_kwarg_doc'):
            return func._kwarg_doc
        docstring = inspect.getdoc(func)
        if docstring is None:
            return 'unknown'
        if docstring.startswith('Alias for '):
            return None
        match = self._get_valid_values_regex.search(docstring)
        if match is not None:
            return re.sub('\n *', ' ', match.group(1))
        param_name = func.__code__.co_varnames[1]
        match = re.search(f'(?m)^ *\\*?{param_name} : (.+)', docstring)
        if match:
            return match.group(1)
        return 'unknown'

    @staticmethod
    @cache
    def is_alias(method):
        """
        Return whether the object *method* is an alias for another method.
        """
        ds = inspect.getdoc(method)
        if ds is None:
            return False
        return ds.startswith('Alias for ')
    _NOT_LINKABLE = {'matplotlib.image._ImageBase.set_alpha', 'matplotlib.image._ImageBase.set_array', 'matplotlib.image._ImageBase.set_data', 'matplotlib.image._ImageBase.set_filternorm', 'matplotlib.image._ImageBase.set_filterrad', 'matplotlib.image._ImageBase.set_interpolation', 'matplotlib.image._ImageBase.set_interpolation_stage', 'matplotlib.image._ImageBase.set_resample', 'matplotlib.text._AnnotationBase.set_annotation_clip'}
