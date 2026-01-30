import re

class _VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""
    _white_spaces = re.compile('\\s\\s+')

    def get_stop_words(self):
        """Build or fetch the effective stop words list.

        Returns
        -------
        stop_words: list or None
                A list of stop words.
        """
        return _check_stop_list(self.stop_words)
