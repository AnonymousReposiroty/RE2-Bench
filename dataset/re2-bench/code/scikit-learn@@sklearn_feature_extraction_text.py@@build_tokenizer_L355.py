import re

class _VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""
    _white_spaces = re.compile('\\s\\s+')

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens.

        Returns
        -------
        tokenizer: callable
              A function to split a string into a sequence of tokens.
        """
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        if token_pattern.groups > 1:
            raise ValueError('More than 1 capturing group in token pattern. Only a single group should be captured.')
        return token_pattern.findall
