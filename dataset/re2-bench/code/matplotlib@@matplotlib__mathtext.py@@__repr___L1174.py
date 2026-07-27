import textwrap
import typing as T

class List(Box):
    """A list of nodes (either horizontal or vertical)."""

    def __init__(self, elements: T.Sequence[Node]):
        super().__init__(0.0, 0.0, 0.0)
        self.shift_amount = 0.0
        self.children = [*elements]
        self.glue_set = 0.0
        self.glue_sign = 0
        self.glue_order = 0

    def __repr__(self):
        return '{}<w={:.02f} h={:.02f} d={:.02f} s={:.02f}>[{}]'.format(super().__repr__(), self.width, self.height, self.depth, self.shift_amount, '\n' + textwrap.indent('\n'.join(map('{!r},'.format, self.children)), '  ') + '\n' if self.children else '')
