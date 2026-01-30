class NamedInitializer(Node):
    __slots__ = ('name', 'expr', 'coord', '__weakref__')

    def __init__(self, name, expr, coord=None):
        self.name = name
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(('expr', self.expr))
        for i, child in enumerate(self.name or []):
            nodelist.append(('name[%d]' % i, child))
        return tuple(nodelist)
    attr_names = ()
