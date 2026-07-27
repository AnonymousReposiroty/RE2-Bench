class For(Node):
    __slots__ = ('init', 'cond', 'next', 'stmt', 'coord', '__weakref__')

    def __init__(self, init, cond, next, stmt, coord=None):
        self.init = init
        self.cond = cond
        self.next = next
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.init is not None:
            nodelist.append(('init', self.init))
        if self.cond is not None:
            nodelist.append(('cond', self.cond))
        if self.next is not None:
            nodelist.append(('next', self.next))
        if self.stmt is not None:
            nodelist.append(('stmt', self.stmt))
        return tuple(nodelist)
    attr_names = ()
