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

    def get_setters(self):
        """
        Get the attribute strings with setters for object.

        For example, for a line, return ``['markerfacecolor', 'linewidth',
        ....]``.
        """
        setters = []
        for name in dir(self.o):
            if not name.startswith('set_'):
                continue
            func = getattr(self.o, name)
            if not callable(func) or self.number_of_parameters(func) < 2 or self.is_alias(func):
                continue
            setters.append(name[4:])
        return setters

    @staticmethod
    @cache
    def number_of_parameters(func):
        """Return number of parameters of the callable *func*."""
        return len(inspect.signature(func).parameters)

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

    def aliased_name(self, s):
        """
        Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME'.

        For example, for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'.
        """
        aliases = ''.join((' or %s' % x for x in sorted(self.aliasd.get(s, []))))
        return s + aliases
    _NOT_LINKABLE = {'matplotlib.image._ImageBase.set_alpha', 'matplotlib.image._ImageBase.set_array', 'matplotlib.image._ImageBase.set_data', 'matplotlib.image._ImageBase.set_filternorm', 'matplotlib.image._ImageBase.set_filterrad', 'matplotlib.image._ImageBase.set_interpolation', 'matplotlib.image._ImageBase.set_interpolation_stage', 'matplotlib.image._ImageBase.set_resample', 'matplotlib.text._AnnotationBase.set_annotation_clip'}

    def pprint_setters(self, prop=None, leadingspace=2):
        """
        If *prop* is *None*, return a list of strings of all settable
        properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of property : valid
        values.
        """
        if leadingspace:
            pad = ' ' * leadingspace
        else:
            pad = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return f'{pad}{prop}: {accepts}'
        lines = []
        for prop in sorted(self.get_setters()):
            accepts = self.get_valid_values(prop)
            name = self.aliased_name(prop)
            lines.append(f'{pad}{name}: {accepts}')
        return lines
