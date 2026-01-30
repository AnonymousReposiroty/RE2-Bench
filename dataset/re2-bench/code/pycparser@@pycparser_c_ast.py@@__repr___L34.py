class Node(object):
    __slots__ = ()
    ' Abstract base class for AST nodes.\n    '

    def __repr__(self):
        """ Generates a python representation of the current node
        """
        result = self.__class__.__name__ + '('
        indent = ''
        separator = ''
        for name in self.__slots__[:-2]:
            result += separator
            result += indent
            result += name + '=' + _repr(getattr(self, name)).replace('\n', '\n  ' + ' ' * (len(name) + len(self.__class__.__name__)))
            separator = ','
            indent = '\n ' + ' ' * len(self.__class__.__name__)
        result += indent + ')'
        return result
