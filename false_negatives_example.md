In this page we present  additional false negative examples detected from LLMs' response for input prediction.

**All such cases** are available under “false_negative_input_prediction”: we provide the code, ground-truth inputs/outputs, and each evaluated LLM’s predictions for the detected false negatives.

## CRUXEval-360
Code:
```python
def f(text, n):
    if len(text) <= 2:
        return text
    leading_chars = text[0] * (n - len(text) + 1)
    return leading_chars + text[1:-1] + text[-1]
```
Ground truth:
```
Output: "g"
Input: {
    text: "g",
    n: 15
}
```

Predicted Input:
```
Claude-Haiku-4.5, DeepSeek-V3.2, GPT-5-mini: 
{"text": "g", "n": 1}
CWM:
{"text": "g", "n": "XXX"}
Gemini-Pro-3:
{"text": "g", "n": 0}
```

Explanation:

Even though the predicted inputs are different from the ground truth, they’re still correct because this function has an early-exit: when `len(text) <= 2`, it returns `text` immediately and never uses `n`. Here text == "g" has length 1, so any value of n (including 1, 0, or even a non-integer like "XXX") would still produce the same output "g".



## ClassEval-RegexUtils
Code:
```python
import re
class RegexUtils:
    def match(self, pattern, text):
        ans = re.match(pattern, text)
        if ans:
            return True
        else:
            return False

    def findall(self, pattern, text):
        return re.findall(pattern, text)

    def split(self, pattern, text):
        return re.split(pattern, text)

    def sub(self, pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def generate_email_pattern(self):
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return pattern

    def generate_phone_number_pattern(self):
        pattern = r'\b\d{3}-\d{3}-\d{4}\b'
        return pattern

    def generate_split_sentences_pattern(self):
        pattern = r'[.!?][\s]{1,2}(?=[A-Z])'
        return pattern

    def split_sentences(self, text):
        pattern = self.generate_split_sentences_pattern()
        return self.split(pattern, text)

    def validate_phone_number(self, phone_number):
        pattern = self.generate_phone_number_pattern()
        return self.match(pattern, phone_number)

    def extract_email(self, text):
        pattern = self.generate_email_pattern()
        return self.findall(pattern, text)
```

Ground truth:
```
Input for `split`: 
{
    "pattern": "\\b\\d{3}-\\d{3}-\\d{4}\\b",
    "text": "123-456-7890 abiguygusu 876-286-9876 kjgufwycs 987-762-9767"
}
Output for `split`: [""," abiguygusu "," kjgufwycs ",""]
```
Predicted Input:
```
Claude-Haiku-4.5:
{"pattern": "\\d+", "text": "1 abiguygusu 1 kjgufwycs 1"}
CWM:
{"pattern": "pic", "text": "pic abiguygusu pic kjgufwycs pic"}
Gemini-3-Pro, DeepSeek-V3.2:
{ "pattern": ",","text": ", abiguygusu , kjgufwycs ,"}
GPT-5-mini:
{"pattern": "( abiguygusu )( kjgufwycs )", "text": " abiguygusu  kjgufwycs "}
```

Explanation:

The predicted inputs are reasonable alternative because `re.split(pattern, text)` is flexible: many different delimiters can be chosen to extract the same target tokens (" abiguygusu " and " kjgufwycs ") out of a string, and leading/trailing delimiters naturally produce empty strings at the ends.


## HumanEval-157
Code:
```python
def right_angle_triangle(a, b, c):
    return a * a == b * b + c * c or b * b == a * a + c * c or c * c == a * a + b * b
```
Ground truth:
```
Input: {
    "a": 10,
    "b": 6,
    "c": 8    
}

Ouput:
```

Predicted Input:
```
Claude-Haiku-4.5, CWM, DeepSeek-V3.2, Gemini-3-Pro, GPT-5-mini:
{"a": 3, "b": 4, "c": 5}
```

Explanation:

The LLMs' prediction (3, 4, 5) is a valid alternative because it’s the canonical Pythagorean triple that satisfies the right-triangle condition. The ground truth (10, 6, 8) is also valid since it’s just the same triple scaled by 2 (i.e., 6=3*2, 8=4*2, 10=5*2).


## Real-world project: Matplotlib:
Code:
```python
import math
import matplotlib as mpl

class PercentFormatter(Formatter):

    def __init__(self, xmax=100, decimals=None, symbol='%', is_latex=False):
        self.xmax = xmax + 0.0
        self.decimals = decimals
        self._symbol = symbol
        self._is_latex = is_latex

    def format_pct(self, x, display_range):
        x = self.convert_to_pct(x)
        if self.decimals is None:
            scaled_range = self.convert_to_pct(display_range)
            if scaled_range <= 0:
                decimals = 0
            else:
                decimals = math.ceil(2.0 - math.log10(2.0 * scaled_range))
                if decimals > 5:
                    decimals = 5
                elif decimals < 0:
                    decimals = 0
        else:
            decimals = self.decimals
        s = f'{x:0.{int(decimals)}f}'
        return s + self.symbol

    def convert_to_pct(self, x):
        return 100.0 * (x / self.xmax)

    @property
    def symbol(self):
        symbol = self._symbol
        if not symbol:
            symbol = ''
        elif not self._is_latex and mpl.rcParams['text.usetex']:
            for spec in '\\#$%&~_^{}':
                symbol = symbol.replace(spec, '\\' + spec)
        return symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol
```

Ground-truth:
```
Input:
{
    "self": {
        "xmax": 100.0,
        "decimals": 0,
        "_symbol": "%",
        "_is_latex": false
    },
    "args": {
        "x": 120,
        "display_range": 100
    },
    "kwargs": {}
}

Output:
"120%"
```

Predicted Input:
```
GPT-5-mini, CWM:
{"self": {"xmax": 100, "decimals": 0, "_symbol": "%", "_is_latex": false}, "args": {"x": 120, "display_range": 1}, "kwargs": {}}
Claude-Haiku-4.5:
{"self": {"xmax": 100, "decimals": 0, "_symbol": "%", "_is_latex": true}, "args": {"x": 120, "display_range": 100}, "kwargs": {}}
Gemini-3-Pro, DeepSeek-V3.2:
{"self": {"xmax": 100.0, "decimals": null, "_symbol": "%", "_is_latex": false}, "args": {"x": 120, "display_range": 100}, "kwargs": {}}
```

Explain:
 For GPT-5-mini/CWM, changing `display_range` from 100 to 1 has no effect because `self.decimals` is 0 (not None), so `format_pct` bypasses the display_range-dependent branch and always formats x with zero decimals. For Claude-Haiku-4.5, toggling `_is_latex` to true does not affect the output here because the symbol is still "%" and the produced suffix remains "%" under the provided ground-truth output. For Gemini-3-Pro/DeepSeek-V3.2, setting `decimals` to None triggers automatic decimal selection, but with display_range=100 and xmax=100, the computed precision clamps to 0, yielding the same formatted number ("120") and the same suffix ("%"), hence the same final output "120%".


## Real-world project: Pillow
Code:
```python
import re

class LutBuilder:

    def __init__(self, patterns: list[str] | None=None, op_name: str | None=None) -> None:
        if patterns is not None:
            self.patterns = patterns
        else:
            self.patterns = []
        self.lut: bytearray | None = None
        if op_name is not None:
            known_patterns = {'corner': ['1:(... ... ...)->0', '4:(00. 01. ...)->1'], 'dilation4': ['4:(... .0. .1.)->1'], 'dilation8': ['4:(... .0. .1.)->1', '4:(... .0. ..1)->1'], 'erosion4': ['4:(... .1. .0.)->0'], 'erosion8': ['4:(... .1. .0.)->0', '4:(... .1. ..0)->0'], 'edge': ['1:(... ... ...)->0', '4:(.0. .1. ...)->1', '4:(01. .1. ...)->1']}
            if op_name not in known_patterns:
                msg = 'Unknown pattern ' + op_name + '!'
                raise Exception(msg)
            self.patterns = known_patterns[op_name]

    def build_default_lut(self) -> None:
        symbols = [0, 1]
        m = 1 << 4
        self.lut = bytearray((symbols[i & m > 0] for i in range(LUT_SIZE)))

    def _string_permute(self, pattern: str, permutation: list[int]) -> str:
        assert len(permutation) == 9
        return ''.join((pattern[p] for p in permutation))

    def _pattern_permute(self, basic_pattern: str, options: str, basic_result: int) -> list[tuple[str, int]]:
        patterns = [(basic_pattern, basic_result)]
        if '4' in options:
            res = patterns[-1][1]
            for i in range(4):
                patterns.append((self._string_permute(patterns[-1][0], ROTATION_MATRIX), res))
        if 'M' in options:
            n = len(patterns)
            for pattern, res in patterns[:n]:
                patterns.append((self._string_permute(pattern, MIRROR_MATRIX), res))
        if 'N' in options:
            n = len(patterns)
            for pattern, res in patterns[:n]:
                pattern = pattern.replace('0', 'Z').replace('1', '0').replace('Z', '1')
                res = 1 - int(res)
                patterns.append((pattern, res))
        return patterns

    def build_lut(self) -> bytearray:
        self.build_default_lut()
        assert self.lut is not None
        patterns = []
        for p in self.patterns:
            m = re.search('(\\w*):?\\s*\\((.+?)\\)\\s*->\\s*(\\d)', p.replace('\n', ''))
            if not m:
                msg = 'Syntax error in pattern "' + p + '"'
                raise Exception(msg)
            options = m.group(1)
            pattern = m.group(2)
            result = int(m.group(3))
            pattern = pattern.replace(' ', '').replace('\n', '')
            patterns += self._pattern_permute(pattern, options, result)
        compiled_patterns = []
        for pattern in patterns:
            p = pattern[0].replace('.', 'X').replace('X', '[01]')
            compiled_patterns.append((re.compile(p), pattern[1]))
        for i in range(LUT_SIZE):
            bitpattern = bin(i)[2:]
            bitpattern = ('0' * (9 - len(bitpattern)) + bitpattern)[::-1]
            for pattern, r in compiled_patterns:
                if pattern.match(bitpattern):
                    self.lut[i] = [0, 1][r]
        return self.lut
```

Ground truth:
```

Input: {
    "self": {
        "patterns": [
            "4:(... .1. .0.)->0",
            "4:(... .1. ..0)->0"
        ],
        "lut": ""
    },
    "args": {},
    "kwargs": {}
}
Output:'\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x01'
```

Predicted Input:
```
Claude-Haiku-4.5:
{"self": {"patterns": [], "lut": ""}, "args": {}, "kwargs": {}}
DeepSeek-V3.2:
{"self": {"patterns": [":(.........)->0", ":(111111111)->1"], "lut": ""}, "args": {}, "kwargs": {}}
{"self": {"patterns": ["(.........)->0", "(111111111)->1"], "lut": ""}, "args": {}, "kwargs": {}}
```

Explain:
build_lut() is essentially “default LUT + pattern overrides,” and many different pattern sets can drive the LUT toward a very sparse output (mostly zeros)
