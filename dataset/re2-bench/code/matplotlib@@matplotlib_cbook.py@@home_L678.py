class _Stack:
    """
    Stack of elements with a movable cursor.

    Mimics home/back/forward in a web browser.
    """

    def __init__(self):
        self._pos = -1
        self._elements = []

    def push(self, o):
        """
        Push *o* to the stack after the current position, and return *o*.

        Discard all later elements.
        """
        self._elements[self._pos + 1:] = [o]
        self._pos = len(self._elements) - 1
        return o

    def home(self):
        """
        Push the first element onto the top of the stack.

        The first element is returned.
        """
        return self.push(self._elements[0]) if self._elements else None
