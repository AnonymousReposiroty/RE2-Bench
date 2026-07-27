class Case(Node):
    __slots__ = ('expr', 'stmts', 'coord', '__weakref__')

    def __init__(self, expr, stmts, coord=None):
        self.expr = expr
        self.stmts = stmts
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(('expr', self.expr))
        for i, child in enumerate(self.stmts or []):
            nodelist.append(('stmts[%d]' % i, child))
        return tuple(nodelist)
    attr_names = ()
