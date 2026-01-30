from sympy.core.sorting import default_sort_key
from sympy.printing.printer import Printer, print_function
from sympy.printing.pretty.stringpict import prettyForm, stringPict

class PrettyPrinter(Printer):
    """Printer, which converts an expression into 2D ASCII-art figure."""
    printmethod = '_pretty'
    _default_settings = {'order': None, 'full_prec': 'auto', 'use_unicode': None, 'wrap_line': True, 'num_columns': None, 'use_unicode_sqrt_char': True, 'root_notation': True, 'mat_symbol_style': 'plain', 'imaginary_unit': 'i', 'perm_cyclic': True}

    def __init__(self, settings=None):
        Printer.__init__(self, settings)
        if not isinstance(self._settings['imaginary_unit'], str):
            raise TypeError("'imaginary_unit' must a string, not {}".format(self._settings['imaginary_unit']))
        elif self._settings['imaginary_unit'] not in ('i', 'j'):
            raise ValueError("'imaginary_unit' must be either 'i' or 'j', not '{}'".format(self._settings['imaginary_unit']))
    _print_RandomSymbol = _print_Symbol
    _print_Infinity = _print_Atom
    _print_NegativeInfinity = _print_Atom
    _print_EmptySet = _print_Atom
    _print_Naturals = _print_Atom
    _print_Naturals0 = _print_Atom
    _print_Integers = _print_Atom
    _print_Rationals = _print_Atom
    _print_Complexes = _print_Atom
    _print_EmptySequence = _print_Atom
    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_seq(self, seq, left=None, right=None, delimiter=', ', parenthesize=lambda x: False, ifascii_nougly=True):
        pforms = []
        for item in seq:
            pform = self._print(item)
            if parenthesize(item):
                pform = prettyForm(*pform.parens())
            if pforms:
                pforms.append(delimiter)
            pforms.append(pform)
        if not pforms:
            s = stringPict('')
        else:
            s = prettyForm(*stringPict.next(*pforms))
        s = prettyForm(*s.parens(left, right, ifascii_nougly=ifascii_nougly))
        return s

    def _print_set(self, s):
        if not s:
            return prettyForm('set()')
        items = sorted(s, key=default_sort_key)
        pretty = self._print_seq(items)
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        return pretty
    _print_bell = _print_bernoulli
