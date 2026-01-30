import re
from functools import partial

class _VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""
    _white_spaces = re.compile('\\s\\s+')

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization.

        Returns
        -------
        preprocessor: callable
              A function to preprocess the text before tokenization.
        """
        if self.preprocessor is not None:
            return self.preprocessor
        if not self.strip_accents:
            strip_accents = None
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' % self.strip_accents)
        return partial(_preprocess, accent_function=strip_accents, lower=self.lowercase)
