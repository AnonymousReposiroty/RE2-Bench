class MovieWriterRegistry:
    """Registry of available writer classes by human readable name."""

    def __init__(self):
        self._registered = dict()

    def is_available(self, name):
        """
        Check if given writer is available by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """
        try:
            cls = self._registered[name]
        except KeyError:
            return False
        return cls.isAvailable()
