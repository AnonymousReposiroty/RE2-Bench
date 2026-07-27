class FuncDef(Node):
    __slots__ = ('decl', 'param_decls', 'body', 'coord', '__weakref__')

    def __init__(self, decl, param_decls, body, coord=None):
        self.decl = decl
        self.param_decls = param_decls
        self.body = body
        self.coord = coord

    def children(self):
        nodelist = []
        if self.decl is not None:
            nodelist.append(('decl', self.decl))
        if self.body is not None:
            nodelist.append(('body', self.body))
        for i, child in enumerate(self.param_decls or []):
            nodelist.append(('param_decls[%d]' % i, child))
        return tuple(nodelist)
    attr_names = ()
