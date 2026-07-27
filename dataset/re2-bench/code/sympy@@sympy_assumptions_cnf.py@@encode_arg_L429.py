class EncodedCNF:
    """
    Class for encoding the CNF expression.
    """

    def __init__(self, data=None, encoding=None):
        if not data and (not encoding):
            data = []
            encoding = {}
        self.data = data
        self.encoding = encoding
        self._symbols = list(encoding.keys())

    def encode_arg(self, arg):
        literal = arg.lit
        value = self.encoding.get(literal, None)
        if value is None:
            n = len(self._symbols)
            self._symbols.append(literal)
            value = self.encoding[literal] = n + 1
        if arg.is_Not:
            return -value
        else:
            return value
