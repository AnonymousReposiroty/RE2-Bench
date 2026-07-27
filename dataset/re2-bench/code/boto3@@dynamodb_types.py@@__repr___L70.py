class Binary:
    """A class for representing Binary in dynamodb

    Especially for Python 2, use this class to explicitly specify
    binary data for item in DynamoDB. It is essentially a wrapper around
    binary. Unicode and Python 3 string types are not allowed.
    """

    def __init__(self, value):
        if not isinstance(value, BINARY_TYPES):
            types = ', '.join([str(t) for t in BINARY_TYPES])
            raise TypeError(f'Value must be of the following types: {types}')
        self.value = value

    def __eq__(self, other):
        if isinstance(other, Binary):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f'Binary({self.value!r})'

    def __str__(self):
        return self.value

    def __bytes__(self):
        return self.value

    def __hash__(self):
        return hash(self.value)
