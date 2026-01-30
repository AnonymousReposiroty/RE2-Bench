import re
import warnings
from functools import partial
import numpy as np

class _VectorizerMixin:
    """Provides common code for text vectorizers (tokenization logic)."""
    _white_spaces = re.compile('\\s\\s+')

    def decode(self, doc):
        """Decode the input into a string of unicode symbols.

        The decoding strategy depends on the vectorizer parameters.

        Parameters
        ----------
        doc : bytes or str
            The string to decode.

        Returns
        -------
        doc: str
            A string of unicode symbols.
        """
        if self.input == 'filename':
            with open(doc, 'rb') as fh:
                doc = fh.read()
        elif self.input == 'file':
            doc = doc.read()
        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)
        if doc is np.nan:
            raise ValueError('np.nan is an invalid document, expected byte or unicode string.')
        return doc

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

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        text_document = self._white_spaces.sub(' ', text_document)
        text_len = len(text_document)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []
        ngrams_append = ngrams.append
        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams_append(text_document[i:i + n])
        return ngrams

    def _char_wb_ngrams(self, text_document):
        """Whitespace sensitive char-n-gram tokenization.

        Tokenize text_document into a sequence of character n-grams
        operating only inside word boundaries. n-grams at the edges
        of words are padded with space."""
        text_document = self._white_spaces.sub(' ', text_document)
        min_n, max_n = self.ngram_range
        ngrams = []
        ngrams_append = ngrams.append
        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in range(min_n, max_n + 1):
                offset = 0
                ngrams_append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset:offset + n])
                if offset == 0:
                    break
        return ngrams

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

    def get_stop_words(self):
        """Build or fetch the effective stop words list.

        Returns
        -------
        stop_words: list or None
                A list of stop words.
        """
        return _check_stop_list(self.stop_words)

    def _check_stop_words_consistency(self, stop_words, preprocess, tokenize):
        """Check if stop words are consistent

        Returns
        -------
        is_consistent : True if stop words are consistent with the preprocessor
                        and tokenizer, False if they are not, None if the check
                        was previously performed, "error" if it could not be
                        performed (e.g. because of the use of a custom
                        preprocessor / tokenizer)
        """
        if id(self.stop_words) == getattr(self, '_stop_words_id', None):
            return None
        try:
            inconsistent = set()
            for w in stop_words or ():
                tokens = list(tokenize(preprocess(w)))
                for token in tokens:
                    if token not in stop_words:
                        inconsistent.add(token)
            self._stop_words_id = id(self.stop_words)
            if inconsistent:
                warnings.warn('Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens %r not in stop_words.' % sorted(inconsistent))
            return not inconsistent
        except Exception:
            self._stop_words_id = id(self.stop_words)
            return 'error'

    def build_analyzer(self):
        """Return a callable to process input data.

        The callable handles preprocessing, tokenization, and n-grams generation.

        Returns
        -------
        analyzer: callable
            A function to handle preprocessing, tokenization
            and n-grams generation.
        """
        if callable(self.analyzer):
            return partial(_analyze, analyzer=self.analyzer, decoder=self.decode)
        preprocess = self.build_preprocessor()
        if self.analyzer == 'char':
            return partial(_analyze, ngrams=self._char_ngrams, preprocessor=preprocess, decoder=self.decode)
        elif self.analyzer == 'char_wb':
            return partial(_analyze, ngrams=self._char_wb_ngrams, preprocessor=preprocess, decoder=self.decode)
        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()
            self._check_stop_words_consistency(stop_words, preprocess, tokenize)
            return partial(_analyze, ngrams=self._word_ngrams, tokenizer=tokenize, preprocessor=preprocess, decoder=self.decode, stop_words=stop_words)
        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' % self.analyzer)
