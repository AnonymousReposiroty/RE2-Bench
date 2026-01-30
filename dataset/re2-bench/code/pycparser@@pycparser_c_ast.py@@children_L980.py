class TernaryOp(Node):
    __slots__ = ('cond', 'iftrue', 'iffalse', 'coord', '__weakref__')

    def __init__(self, cond, iftrue, iffalse, coord=None):
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None:
            nodelist.append(('cond', self.cond))
        if self.iftrue is not None:
            nodelist.append(('iftrue', self.iftrue))
        if self.iffalse is not None:
            nodelist.append(('iffalse', self.iffalse))
        return tuple(nodelist)
    attr_names = ()
