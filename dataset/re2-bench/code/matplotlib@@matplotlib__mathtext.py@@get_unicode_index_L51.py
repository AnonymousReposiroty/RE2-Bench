from ._mathtext_data import (
    latex_to_bakoma, stix_glyph_fixes, stix_virtual_fonts, tex2uni)

def get_unicode_index(symbol: str) -> int:  # Publicly exported.
    r"""
    Return the integer index (from the Unicode table) of *symbol*.

    Parameters
    ----------
    symbol : str
        A single (Unicode) character, a TeX command (e.g. r'\pi') or a Type1
        symbol name (e.g. 'phi').
    """
    try:  # This will succeed if symbol is a single Unicode char
        return ord(symbol)
    except TypeError:
        pass
    try:  # Is symbol a TeX symbol (i.e. \alpha)
        return tex2uni[symbol.strip("\\")]
    except KeyError as err:
        raise ValueError(
            f"{symbol!r} is not a valid Unicode character or TeX/Type1 symbol"
            ) from err
