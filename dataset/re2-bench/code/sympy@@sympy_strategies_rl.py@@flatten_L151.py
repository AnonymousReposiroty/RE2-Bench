from .util import new

def flatten(expr, new=new):
    """ Flatten T(a, b, T(c, d), T2(e)) to T(a, b, c, d, T2(e)) """
    cls = expr.__class__
    args = []
    for arg in expr.args:
        if arg.__class__ == cls:
            args.extend(arg.args)
        else:
            args.append(arg)
    return new(expr.__class__, *args)
