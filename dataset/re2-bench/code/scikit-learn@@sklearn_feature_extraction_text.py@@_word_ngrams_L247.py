import re

class _VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""
    _white_spaces = re.compile('\\s\\s+')

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []
            n_original_tokens = len(original_tokens)
            tokens_append = tokens.append
            space_join = ' '.join
            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i:i + n]))
        return tokens
